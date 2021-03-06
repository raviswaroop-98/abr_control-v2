"""
Running sliding control with the PyGame display. The arm will
move the end-effector to the target, which can be moved by
clicking on the background.
"""
from abr_control.arms import threejoint as arm
from abr_control.controllers import Sliding

# from abr_control.arms import twojoint as arm
from abr_control.interfaces.pygame import PyGame

# initialize our robot config
robot_config = arm.Config()
# create our arm simulation
arm_sim = arm.ArmSim(robot_config)

# create an operational space controller
ctrlr = Sliding(robot_config)


def on_click(self, mouse_x, mouse_y):
    self.target[0] = self.mouse_x
    self.target[1] = self.mouse_y


# create our interface
interface = PyGame(robot_config, arm_sim, dt=0.001, on_click=on_click)
interface.connect()

# create a target
feedback = interface.get_feedback()
target_xyz = robot_config.Tx("EE", feedback["q"])
interface.set_target(target_xyz)


try:
    print("\nSimulation starting...")
    print("Click to move the target.\n")

    count = 0
    while 1:
        # get arm feedback
        feedback = interface.get_feedback()
        hand_xyz = robot_config.Tx("EE", feedback["q"])

        # generate an operational space control signal
        u = ctrlr.generate(q=feedback["q"], dq=feedback["dq"], target=target_xyz)

        new_target = interface.get_mousexy()
        if new_target is not None:
            target_xyz[0:2] = new_target
        interface.set_target(target_xyz)

        # apply the control signal, step the sim forward
        interface.send_forces(u, update_display=(count % 20 == 0))

        count += 1

finally:
    # stop and reset the simulation
    interface.disconnect()

    print("Simulation terminated...")
