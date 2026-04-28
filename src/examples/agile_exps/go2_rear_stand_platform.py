import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import meshcat.geometry as g
import meshcat.transformations as tf

from nltrajopt.trajectory_optimization import NLTrajOpt
from nltrajopt.contact_scheduler import ContactScheduler
from nltrajopt.node import Node
from nltrajopt.constraint_models import *
from nltrajopt.constraint_models.abstract_constraint import extend_ids_lists
from nltrajopt.cost_models import *
import nltrajopt.utils as reprutils

from terrain.terrain_grid import TerrainGrid
from robots.go2.Go2Wrapper import Go2
from visualiser.visualiser import TrajoptVisualiser

import nltrajopt.params as pars


VIS = pars.VIS
DT = 0.1
PLAYBACK_SLOWDOWN = 3.0

# Terrain / platform geometry.
PLATFORM_HEIGHT = 0.5
PLATFORM_X_MIN = 0.5
PLATFORM_X_MAX = 1.2
PLATFORM_Y_MIN = -0.8
PLATFORM_Y_MAX = 0.8
PLATFORM_STL_PATH = Path(__file__).parent / "assets" / "front_platform.stl"

# Contact phase timing (seconds).
DOUBLE_SUPPORT_START = 0.5
REAR_ONLY_LIFT = 0.5
DOUBLE_SUPPORT_END = 0.5

# Base orientation targets (radians).
STAND_PITCH = -1.3
TARGET_PITCH = -1.15

# Cost weights.
BASE_SYMMETRY_WEIGHTS = [1e-2, 1e-1, 1e-1]
JOINT_MIRROR_WEIGHTS = [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]
Q_REG_WEIGHT = 1e-6
QDD_REG_WEIGHT = 1e-7
ACTIVE_Q_WEIGHT = 1e-1

# Swing shaping / contact approach tuning.
FRONT_REACH_RATIO = 0.85
STAND_BLEND_RATIO = 0.6
FRONT_FOOT_SWING_MAX_CLEARANCE = 0.12
PRE_CONTACT_WARMUP_NODES = 2


def set_base_rpy(q, rpy):
    q_out = np.copy(q)
    q_out[3:7] = pin.Quaternion(pin.rpy.rpyToMatrix(*rpy)).coeffs()
    return q_out


def add_front_platform(terrain):
    x_range = np.linspace(terrain.min_x, terrain.max_x, terrain.rows)
    y_range = np.linspace(terrain.min_y, terrain.max_y, terrain.cols)
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            if PLATFORM_X_MIN <= x <= PLATFORM_X_MAX and PLATFORM_Y_MIN <= y <= PLATFORM_Y_MAX:
                terrain.grid[i, j] = PLATFORM_HEIGHT


def load_platform_stl(tvis):
    center_x = 0.5 * (PLATFORM_X_MIN + PLATFORM_X_MAX)
    center_y = 0.5 * (PLATFORM_Y_MIN + PLATFORM_Y_MAX)
    center_z = 0.5 * PLATFORM_HEIGHT
    tvis.vis.viewer["terrain"].delete()
    tvis.vis.viewer["terrain"]["front_platform"].set_object(
        g.StlMeshGeometry.from_file(str(PLATFORM_STL_PATH)),
        g.MeshLambertMaterial(color=0xDDDDDD),
    )
    tvis.vis.viewer["terrain"]["front_platform"].set_transform(
        tf.translation_matrix([center_x, center_y, center_z])
    )


def find_contact_transition(frame_contact_seq, probe_frame, to_contact):
    return next(
        k
        for k, contact_phase_fnames in enumerate(frame_contact_seq)
        if k > 0
        and ((probe_frame in contact_phase_fnames) == to_contact)
        and ((probe_frame in frame_contact_seq[k - 1]) != to_contact)
    )


def mean_frame_translation(robot, frame_names, q, dim=None):
    robot.fk_all(q)
    values = [
        robot.data.oMf[robot.model.getFrameId(frame)].translation
        for frame in frame_names
    ]
    mean_val = np.mean(values, axis=0)
    return mean_val if dim is None else mean_val[:dim]


def align_pose_to_rear_support(robot, q_ref_xy, q):
    rear_frames = robot.left_foot_frames + robot.right_foot_frames
    rear_foot_pos = mean_frame_translation(robot, rear_frames, q)
    q[0] += q_ref_xy[0] - rear_foot_pos[0]
    q[1] += q_ref_xy[1] - rear_foot_pos[1]
    q[2] -= rear_foot_pos[2]


class BaseSymmetryCost:
    def __init__(self, weights):
        self.ids = [1, 3, 5]  # lateral translation, roll, yaw in SE(3) tangent coordinates
        self.weights = np.asarray(weights)

    def obj(self, opt_vect, node, next_node=None):
        values = opt_vect[node.q_id][self.ids]
        return 0.5 * np.sum(self.weights * values**2)

    def grad(self, opt_vect, cost_grad, node, next_node=None):
        values = opt_vect[node.q_id][self.ids]
        cost_grad[node.q_id.start + np.asarray(self.ids)] += self.weights * values


class JointMirrorSymmetryCost:
    def __init__(self, weights):
        self.weights = np.asarray(weights)

    def _residual(self, opt_vect, node):
        qj = opt_vect[node.q_id][6:]
        return np.array(
            [
                qj[0] + qj[3],
                qj[1] - qj[4],
                qj[2] - qj[5],
                qj[6] + qj[9],
                qj[7] - qj[10],
                qj[8] - qj[11],
            ]
        )

    def obj(self, opt_vect, node, next_node=None):
        res = self._residual(opt_vect, node)
        return 0.5 * np.sum(self.weights * res**2)

    def grad(self, opt_vect, cost_grad, node, next_node=None):
        weighted_res = self.weights * self._residual(opt_vect, node)
        q_start = node.q_id.start + 6
        cost_grad[q_start + 0] += weighted_res[0]
        cost_grad[q_start + 3] += weighted_res[0]
        cost_grad[q_start + 1] += weighted_res[1]
        cost_grad[q_start + 4] -= weighted_res[1]
        cost_grad[q_start + 2] += weighted_res[2]
        cost_grad[q_start + 5] -= weighted_res[2]
        cost_grad[q_start + 6] += weighted_res[3]
        cost_grad[q_start + 9] += weighted_res[3]
        cost_grad[q_start + 7] += weighted_res[4]
        cost_grad[q_start + 10] -= weighted_res[4]
        cost_grad[q_start + 8] += weighted_res[5]
        cost_grad[q_start + 11] -= weighted_res[5]


class ActiveConfigurationCost:
    def __init__(self, ref, weight, active_from_k):
        self.ref = ref.reshape(-1, 1)
        self.Q = weight
        self.active_from_k = active_from_k

    def obj(self, opt_vect, node, next_node=None):
        if node.k < self.active_from_k:
            return 0.0
        var = opt_vect[node.q_id.start + pars.SPACE_NQ : node.q_id.stop].reshape(-1, 1)
        res = var - self.ref
        return 0.5 * np.sum(res.T @ self.Q @ res)

    def grad(self, opt_vect, cost_grad, node, next_node=None):
        if node.k < self.active_from_k:
            return
        var = opt_vect[node.q_id.start + pars.SPACE_NQ : node.q_id.stop].reshape(-1, 1)
        res = var - self.ref
        jac = (res.T @ self.Q).reshape((-1,))
        cost_grad[node.q_id.start + pars.SPACE_NQ : node.q_id.stop] += jac


class FramePlatformFrontClearanceConstraint:
    def __init__(self, frame_names, max_x, min_z, active_from_k):
        self.frame_names = frame_names
        self.max_x = max_x
        self.min_z = min_z
        self.active_from_k = active_from_k

    @property
    def name(self):
        return "frame_platform_front_clearance"

    def init_constraint_ids(self, node):
        prev_slice = node.c_extra_last_id
        node.extra_constraint_ids[self.name] = {}
        for frame_name in self.frame_names:
            node.extra_constraint_ids[self.name][frame_name] = slice(prev_slice.stop, prev_slice.stop + 2)
            prev_slice = node.extra_constraint_ids[self.name][frame_name]
            node.c_dim += 2
        node.c_extra_last_id = prev_slice

    def compute_constraints(self, node_curr, node_next, state_vars, c, model, data):
        if node_curr.k < self.active_from_k:
            return

        q = reprutils.rep2pin(state_vars[node_curr.q_id])
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for frame_name in self.frame_names:
            pos = data.oMf[model.getFrameId(frame_name)].translation
            c_ids = node_curr.extra_constraint_ids[self.name][frame_name]
            c[c_ids] = [self.max_x - pos[0], pos[2] - self.min_z]

    def compute_jacobians(self, node_curr, node_next, w, jac, model, data):
        if node_curr.k < self.active_from_k:
            return

        q = reprutils.rep2pin(w[node_curr.q_id])
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for frame_name in self.frame_names:
            frame_id = model.getFrameId(frame_name)
            J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
            J[:, :6] = J[:, :6] @ pin.Jexp6(w[node_curr.q_id][:6])
            c_ids = node_curr.extra_constraint_ids[self.name][frame_name]
            jac[c_ids.start, node_curr.q_id] = -J[0, :]
            jac[c_ids.start + 1, node_curr.q_id] = J[2, :]

    def get_structure_ids(self, node_curr, node_next, row_ids, col_ids):
        for frame_name in self.frame_names:
            extend_ids_lists(row_ids, col_ids, node_curr.extra_constraint_ids[self.name][frame_name], node_curr.q_id)

    def get_bounds(self, node, lb, ub, clb, cub, model):
        for frame_name in self.frame_names:
            c_ids = node.extra_constraint_ids[self.name][frame_name]
            if node.k < self.active_from_k:
                clb[c_ids] = [0.0, 0.0]
                cub[c_ids] = [0.0, 0.0]
            else:
                cub[c_ids] = [None, None]


class FrameHeightUpperBoundConstraint:
    def __init__(self, frame_names, max_z, active_from_k, active_until_k):
        self.frame_names = frame_names
        self.max_z = max_z
        self.active_from_k = active_from_k
        self.active_until_k = active_until_k

    @property
    def name(self):
        return "frame_height_upper_bound"

    def _is_active(self, k):
        return self.active_from_k <= k < self.active_until_k

    def init_constraint_ids(self, node):
        prev_slice = node.c_extra_last_id
        node.extra_constraint_ids[self.name] = {}
        for frame_name in self.frame_names:
            node.extra_constraint_ids[self.name][frame_name] = slice(prev_slice.stop, prev_slice.stop + 1)
            prev_slice = node.extra_constraint_ids[self.name][frame_name]
            node.c_dim += 1
        node.c_extra_last_id = prev_slice

    def compute_constraints(self, node_curr, node_next, state_vars, c, model, data):
        if not self._is_active(node_curr.k):
            return
        q = reprutils.rep2pin(state_vars[node_curr.q_id])
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for frame_name in self.frame_names:
            pos = data.oMf[model.getFrameId(frame_name)].translation
            c_ids = node_curr.extra_constraint_ids[self.name][frame_name]
            c[c_ids] = [self.max_z - pos[2]]

    def compute_jacobians(self, node_curr, node_next, w, jac, model, data):
        if not self._is_active(node_curr.k):
            return
        q = reprutils.rep2pin(w[node_curr.q_id])
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for frame_name in self.frame_names:
            frame_id = model.getFrameId(frame_name)
            J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
            J[:, :6] = J[:, :6] @ pin.Jexp6(w[node_curr.q_id][:6])
            c_ids = node_curr.extra_constraint_ids[self.name][frame_name]
            jac[c_ids.start, node_curr.q_id] = -J[2, :]

    def get_structure_ids(self, node_curr, node_next, row_ids, col_ids):
        for frame_name in self.frame_names:
            extend_ids_lists(row_ids, col_ids, node_curr.extra_constraint_ids[self.name][frame_name], node_curr.q_id)

    def get_bounds(self, node, lb, ub, clb, cub, model):
        for frame_name in self.frame_names:
            c_ids = node.extra_constraint_ids[self.name][frame_name]
            if not self._is_active(node.k):
                clb[c_ids] = [0.0]
                cub[c_ids] = [0.0]
            else:
                cub[c_ids] = [None]


terrain = TerrainGrid(40, 40, 0.9, -1.0, -5.0, 5.0, 5.0)
terrain.set_zero()
add_front_platform(terrain)

robot = Go2()
q0 = robot.go_neutral()

contacts_dict = {
    "rear_feet": robot.left_foot_frames + robot.right_foot_frames,
    "front_feet": robot.left_gripper_frames + robot.right_gripper_frames,
    "RL": robot.left_foot_frames,
    "RR": robot.right_foot_frames,
    "FL": robot.left_gripper_frames,
    "FR": robot.right_gripper_frames,
}

contact_scheduler = ContactScheduler(robot.model, dt=DT, contact_frame_dict=contacts_dict)

contact_scheduler.add_phase(["rear_feet", "front_feet"], DOUBLE_SUPPORT_START)
contact_scheduler.add_phase(["rear_feet"], REAR_ONLY_LIFT)
contact_scheduler.add_phase(["rear_feet", "front_feet"], DOUBLE_SUPPORT_END)

frame_contact_seq = contact_scheduler.contact_sequence_fnames
print("K = ", len(frame_contact_seq))
print(
    f"Front platform: height={PLATFORM_HEIGHT}m, "
    f"x=[{PLATFORM_X_MIN}, {PLATFORM_X_MAX}], y=[{PLATFORM_Y_MIN}, {PLATFORM_Y_MAX}]"
)
contact_frame_names = (
    robot.left_foot_frames
    + robot.right_foot_frames
    + robot.left_gripper_frames
    + robot.right_gripper_frames
)
probe_front_frame = robot.left_gripper_frames[0]
platform_contact_start = find_contact_transition(
    frame_contact_seq, probe_front_frame, to_contact=True
)
front_liftoff_start = find_contact_transition(
    frame_contact_seq, probe_front_frame, to_contact=False
)

stages = []
for contact_phase_fnames in frame_contact_seq:
    stage_node = Node(
        nv=robot.model.nv,
        contact_phase_fnames=contact_phase_fnames,
        contact_fnames=contact_frame_names,
    )

    dyn_const = WholeBodyDynamics()
    stage_node.dynamics_type = dyn_const.name
    stage_node.constraints_list.extend(
        [
            dyn_const,
            TimeConstraint(min_dt=DT, max_dt=DT, total_time=None),
            SemiEulerIntegration(),
            TerrainGridContactConstraints(terrain),
            TerrainGridFrictionConstraints(terrain, max_delta_force=200.0),
            FramePlatformFrontClearanceConstraint(
                ["FL_calf_joint", "FR_calf_joint"],
                max_x=PLATFORM_X_MIN - 0.03,
                min_z=PLATFORM_HEIGHT,
                active_from_k=max(0, platform_contact_start - PRE_CONTACT_WARMUP_NODES),
            ),
            FrameHeightUpperBoundConstraint(
                robot.left_gripper_frames + robot.right_gripper_frames,
                max_z=PLATFORM_HEIGHT + FRONT_FOOT_SWING_MAX_CLEARANCE,
                active_from_k=front_liftoff_start,
                active_until_k=platform_contact_start,
            ),
        ]
    )
    stage_node.costs_list.extend(
        [
            BaseSymmetryCost(BASE_SYMMETRY_WEIGHTS),
            JointMirrorSymmetryCost(JOINT_MIRROR_WEIGHTS),
            ConfigurationCost(q0.copy()[7:], np.eye(robot.model.nq - 7) * Q_REG_WEIGHT),
            JointAccelerationCost(
                np.zeros((robot.model.nv - 6,)),
                np.eye(robot.model.nv - 6) * QDD_REG_WEIGHT,
            ),
        ]
    )
    stages.append(stage_node)

opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)

opti.set_initial_pose(q0)

q_stand = set_base_rpy(q0, [0.0, STAND_PITCH, 0.0])
qf = set_base_rpy(q0, [0.0, TARGET_PITCH, 0.0])

# Fold the rear legs under the body and tuck the front legs. This keeps the
# center of mass close to the rear-foot support line for a static final pose.
q_stand[8] = 1.4
q_stand[9] = -2.2
q_stand[11] = 1.4
q_stand[12] = -2.2
q_stand[14] = 2.264496275231389
q_stand[15] = -2.033333333333333
q_stand[17] = 2.264496275231389
q_stand[18] = -2.033333333333333

# Final posture: rear feet keep their original ground contact while both front
# feet reach the nearby 0.5 m front terrain.
qf[8] = 0.5
qf[9] = -0.9104310393851263
qf[11] = 0.5
qf[12] = -0.9104310393851263
qf[14] = 2.2
qf[15] = -1.0
qf[17] = 2.2
qf[18] = -1.0

# Make the stand waypoint a short transitional posture instead of a long
# vertical hold, so the front legs can approach the platform earlier.
q_stand[7:] = (1.0 - STAND_BLEND_RATIO) * q_stand[7:] + STAND_BLEND_RATIO * qf[7:]

# Keep front-feet swing low and close to the final contact posture to avoid
# lifting too high and "slapping" onto the platform.
for idx in [14, 15, 17, 18]:
    q_stand[idx] = (1.0 - FRONT_REACH_RATIO) * q_stand[idx] + FRONT_REACH_RATIO * qf[idx]

rear_foot_xy = mean_frame_translation(
    robot,
    robot.left_foot_frames + robot.right_foot_frames,
    q0,
    dim=2,
)
for q in (q_stand, qf):
    align_pose_to_rear_support(robot, rear_foot_xy, q)

front_foot_mean_z = mean_frame_translation(
    robot,
    robot.left_gripper_frames + robot.right_gripper_frames,
    qf,
)[2]
qf[2] += PLATFORM_HEIGHT - front_foot_mean_z
opti.set_target_pose(qf)

for node in opti.nodes:
    node.costs_list.append(
        ActiveConfigurationCost(
            qf.copy()[7:],
            np.eye(robot.model.nq - 7) * ACTIVE_Q_WEIGHT,
            active_from_k=max(0, platform_contact_start - PRE_CONTACT_WARMUP_NODES),
        )
    )

swing_start = int((DOUBLE_SUPPORT_START + 0.2) / DT)
for k, node in enumerate(opti.nodes):
    if k <= swing_start:
        alpha = k / swing_start
        q_start = q0
        q_goal = q_stand
        pitch = STAND_PITCH
        pitch_start = 0.0
    else:
        transition_steps = max(1, platform_contact_start - swing_start)
        alpha = min(1.0, (k - swing_start) / transition_steps)
        q_start = q_stand
        q_goal = qf
        pitch = TARGET_PITCH
        pitch_start = STAND_PITCH
    # Use smootherstep to reduce approach velocity near the contact transition.
    smooth = 6 * alpha**5 - 15 * alpha**4 + 10 * alpha**3
    q_guess = np.copy(q0)
    q_guess[:3] = (1.0 - smooth) * q_start[:3] + smooth * q_goal[:3]
    q_guess[7:] = (1.0 - smooth) * q_start[7:] + smooth * q_goal[7:]
    opti.x0[node.q_id] = reprutils.rpy2rep(
        q_guess,
        [0.0, (1.0 - smooth) * pitch_start + smooth * pitch, 0.0],
    )

result = opti.solve(500, 1e-3, parallel=False, print_level=5)
opti.save_solution("go2_rear_stand_platform")

K = len(result["nodes"])
dts = [result["nodes"][k]["dt"] for k in range(K)]
qs = [result["nodes"][k]["q"] for k in range(K)]
forces = [result["nodes"][k]["forces"] for k in range(K)]

if VIS:
    tvis = TrajoptVisualiser(robot)
    tvis.display_robot_q(robot, qs[0])
    load_platform_stl(tvis)

    time.sleep(1)
    while True:
        for i in range(len(qs)):
            playback_dt = dts[i] * PLAYBACK_SLOWDOWN
            time.sleep(playback_dt)
            tvis.display_robot_q(robot, qs[i])
            tvis.update_forces(robot, forces[i], 0.01)
        tvis.update_forces(robot, {}, 0.01)
