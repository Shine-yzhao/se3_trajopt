import time

import numpy as np
import pinocchio as pin

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


def set_base_rpy(q, rpy):
    q_out = np.copy(q)
    q_out[3:7] = pin.Quaternion(pin.rpy.rpyToMatrix(*rpy)).coeffs()
    return q_out


terrain = TerrainGrid(10, 10, 0.9, -1.0, -5.0, 5.0, 5.0)
terrain.set_zero()

robot = Go2()
q0 = robot.go_neutral()

contacts_dict = {
    "rear_feet": robot.left_foot_frames + robot.right_foot_frames,
    "front_feet": robot.left_gripper_frames + robot.right_gripper_frames,
}

contact_scheduler = ContactScheduler(robot.model, dt=DT, contact_frame_dict=contacts_dict)

contact_scheduler.add_phase(["rear_feet", "front_feet"], 0.5)
contact_scheduler.add_phase(["rear_feet"], 1.2)
contact_scheduler.add_phase(["rear_feet"], 0.5)

frame_contact_seq = contact_scheduler.contact_sequence_fnames
print("K = ", len(frame_contact_seq))
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
            ConfigurationCost(q0.copy()[7:], np.eye(robot.model.nq - 7) * 1e-6),
            JointAccelerationCost(np.zeros((robot.model.nv - 6,)), np.eye(robot.model.nv - 6) * 1e-7),
        ]
    )
    stages.append(stage_node)

opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)

opti.set_initial_pose(q0)

target_pitch = -1.4
qf = set_base_rpy(q0, [0.0, target_pitch, 0.0])

# Fold the rear legs under the body and tuck the front legs. This keeps the
# center of mass close to the rear-foot support line for a static final pose.
qf[8] = 1.4
qf[9] = -2.2
qf[11] = 1.4
qf[12] = -2.2
qf[14] = 2.25
qf[15] = -2.03
qf[17] = 2.25
qf[18] = -2.03

robot.fk_all(qf)
rear_foot_heights = [
    robot.data.oMf[robot.model.getFrameId(frame)].translation[2]
    for frame in robot.left_foot_frames + robot.right_foot_frames
]
qf[2] -= np.mean(rear_foot_heights)
opti.set_target_pose(qf)

for k, node in enumerate(opti.nodes):
    alpha = k / (len(opti.nodes) - 1)
    smooth = 3 * alpha**2 - 2 * alpha**3
    q_guess = np.copy(q0)
    q_guess[:3] = (1.0 - smooth) * q0[:3] + smooth * qf[:3]
    q_guess[7:] = (1.0 - smooth) * q0[7:] + smooth * qf[7:]
    opti.x0[node.q_id] = reprutils.rpy2rep(q_guess, [0.0, target_pitch * smooth, 0.0])

result = opti.solve(200, 1e-3, parallel=False, print_level=0)
opti.save_solution("go2_rear_stand")

K = len(result["nodes"])
dts = [result["nodes"][k]["dt"] for k in range(K)]
qs = [result["nodes"][k]["q"] for k in range(K)]
forces = [result["nodes"][k]["forces"] for k in range(K)]

if VIS:
    tvis = TrajoptVisualiser(robot)
    tvis.display_robot_q(robot, qs[0])
    tvis.load_terrain(terrain)

    time.sleep(1)
    while True:
        for i in range(len(qs)):
            time.sleep(dts[i])
            tvis.display_robot_q(robot, qs[i])
            tvis.update_forces(robot, forces[i], 0.01)
        tvis.update_forces(robot, {}, 0.01)
