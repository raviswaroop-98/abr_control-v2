"""
Running operational space control using CoppeliaSim. The controller will
move the end-effector to the target object's orientation.
"""
import numpy as np

from abr_control.arms import ur5 as arm

# from abr_control.arms import jaco2 as arm
from abr_control.controllers import OSC, Damping
from abr_control.interfaces import CoppeliaSim
from abr_control.utils import transformations

# initialize our robot config
robot_config = arm.Config()

# damp the movements of the arm
damping = Damping(robot_config, kv=10)
# create opreational space controller
ctrlr = OSC(
    robot_config,
    kp=200,  # position gain
    ko=200,  # orientation gain
    null_controllers=[damping],
    # control (alpha, beta, gamma) out of [x, y, z, alpha, beta, gamma]
    ctrlr_dof=[False, False, False, True, True, True],
)

# create our interface
interface = CoppeliaSim(robot_config, dt=0.005)
interface.connect()

# set up lists for tracking data
ee_angles_track = []
target_angles_track = []


try:
    print("\nSimulation starting...\n")
    while 1:
        # get arm feedback
        feedback = interface.get_feedback()
        hand_xyz = robot_config.Tx("EE", feedback["q"])

        target = np.hstack(
            [interface.get_xyz("target"), interface.get_orientation("target")]
        )

        rc_matrix = robot_config.R("EE", feedback["q"])
        rc_angles = transformations.euler_from_matrix(rc_matrix, axes="rxyz")

        u = ctrlr.generate(
            q=feedback["q"],
            dq=feedback["dq"],
            target=target,
        )

        # apply the control signal, step the sim forward
        interface.send_forces(u)

        # track data
        ee_angles_track.append(
            transformations.euler_from_matrix(
                robot_config.R("EE", feedback["q"]), axes="rxyz"
            )
        )
        target_angles_track.append(interface.get_orientation("target"))

finally:
    # stop and reset the simulation
    interface.disconnect()

    print("Simulation terminated...")

    ee_angles_track = np.array(ee_angles_track)
    target_angles_track = np.array(target_angles_track)

    if ee_angles_track.shape[0] > 0:
        # plot distance from target and 3D trajectory
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(ee_angles_track)
        plt.gca().set_prop_cycle(None)
        plt.plot(target_angles_track, "--")
        plt.ylabel("3D orientation (rad)")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.show()
