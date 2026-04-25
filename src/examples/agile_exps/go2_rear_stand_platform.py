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
from nltrajopt.cost_models import *
import nltrajopt.utils as reprutils

from terrain.terrain_grid import TerrainGrid
from robots.go2.Go2Wrapper import Go2
from visualiser.visualiser import TrajoptVisualiser

import nltrajopt.params as pars


VIS = pars.VIS
DT = 0.1
HOLD_TIME = 5.0
PLAYBACK_SLOWDOWN = 3.0
PLATFORM_HEIGHT = 0.5
PLATFORM_X_MIN = 0.5
PLATFORM_X_MAX = 1.2
PLATFORM_Y_MIN = -0.8
PLATFORM_Y_MAX = 0.8
PLATFORM_STL_PATH = Path(__file__).parent / "assets" / "front_platform.stl"


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


terrain = TerrainGrid(40, 40, 0.9, -1.0, -5.0, 5.0, 5.0)
terrain.set_zero()
add_front_platform(terrain)

robot = Go2()
q0 = robot.go_neutral()

contacts_dict = {
    "rear_feet": robot.left_foot_frames + robot.right_foot_frames,
    "front_feet": robot.left_gripper_frames + robot.right_gripper_frames,
}

contact_scheduler = ContactScheduler(robot.model, dt=DT, contact_frame_dict=contacts_dict)

contact_scheduler.add_phase(["rear_feet", "front_feet"], 0.5)
contact_scheduler.add_phase(["rear_feet"], 1.2)
contact_scheduler.add_phase(["rear_feet", "front_feet"], 1.0)

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
            TerrainGridFrictionConstraints(terrain, max_delta_force=80.0),
        ]
    )
    stage_node.costs_list.extend(
        [
            BaseSymmetryCost([1e-2, 1e-1, 1e-1]),
            JointMirrorSymmetryCost([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]),
            ConfigurationCost(q0.copy()[7:], np.eye(robot.model.nq - 7) * 1e-6),
            JointAccelerationCost(np.zeros((robot.model.nv - 6,)), np.eye(robot.model.nv - 6) * 1e-7),
        ]
    )
    stages.append(stage_node)

opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)

opti.set_initial_pose(q0)

stand_pitch = -1.4
target_pitch = -0.6
q_stand = set_base_rpy(q0, [0.0, stand_pitch, 0.0])
qf = set_base_rpy(q0, [0.0, target_pitch, 0.0])

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
qf[8] = -0.5
qf[9] = -1.67
qf[11] = -0.5
qf[12] = -1.67
qf[14] = 2.264496275231389
qf[15] = -2.033333333333333
qf[17] = 2.264496275231389
qf[18] = -2.033333333333333

robot.fk_all(q0)
rear_foot_xy = np.mean(
    [
        robot.data.oMf[robot.model.getFrameId(frame)].translation[:2]
        for frame in robot.left_foot_frames + robot.right_foot_frames
    ],
    axis=0,
)
for q in (q_stand, qf):
    robot.fk_all(q)
    rear_foot_pos = np.mean(
        [
            robot.data.oMf[robot.model.getFrameId(frame)].translation
            for frame in robot.left_foot_frames + robot.right_foot_frames
        ],
        axis=0,
    )
    q[0] += rear_foot_xy[0] - rear_foot_pos[0]
    q[1] += rear_foot_xy[1] - rear_foot_pos[1]
    q[2] -= rear_foot_pos[2]

robot.fk_all(qf)
front_foot_heights = [
    robot.data.oMf[robot.model.getFrameId(frame)].translation[2]
    for frame in robot.left_gripper_frames + robot.right_gripper_frames
]
qf[2] += PLATFORM_HEIGHT - np.mean(front_foot_heights)
opti.set_target_pose(qf)

stand_end = int((0.5 + 1.2) / DT)
for k, node in enumerate(opti.nodes):
    if k <= stand_end:
        alpha = k / stand_end
        q_start = q0
        q_goal = q_stand
        pitch = stand_pitch
    else:
        alpha = (k - stand_end) / (len(opti.nodes) - 1 - stand_end)
        q_start = q_stand
        q_goal = qf
        pitch = target_pitch
    smooth = 3 * alpha**2 - 2 * alpha**3
    q_guess = np.copy(q0)
    q_guess[:3] = (1.0 - smooth) * q_start[:3] + smooth * q_goal[:3]
    q_guess[7:] = (1.0 - smooth) * q_start[7:] + smooth * q_goal[7:]
    pitch_start = 0.0 if k <= stand_end else stand_pitch
    opti.x0[node.q_id] = reprutils.rpy2rep(
        q_guess,
        [0.0, (1.0 - smooth) * pitch_start + smooth * pitch, 0.0],
    )

result = opti.solve(200, 1e-3, parallel=False, print_level=0)

hold_steps = int(round(HOLD_TIME / DT))
hold_node = result["nodes"][-1]
for _ in range(hold_steps):
    result["nodes"].append(
        {
            "dt": DT,
            "q": hold_node["q"].copy(),
            "v": np.zeros_like(hold_node["v"]),
            "a": np.zeros_like(hold_node["a"]),
            "forces": {frame: force.copy() for frame, force in hold_node["forces"].items()},
            "contact_positions": {
                frame: pos.copy() for frame, pos in hold_node["contact_positions"].items()
            },
        }
    )
opti.sol_dict["nodes"] = result["nodes"]
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
            playback_dt = dts[i] * (PLAYBACK_SLOWDOWN if i < K - hold_steps else 1.0)
            time.sleep(playback_dt)
            tvis.display_robot_q(robot, qs[i])
            tvis.update_forces(robot, forces[i], 0.01)
        tvis.update_forces(robot, {}, 0.01)
