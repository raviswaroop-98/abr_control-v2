import numpy as np
import pyximport

pyximport.install(inplace=True)

from .arm_files.py3LinkArm import pySim  # pylint: disable=C0413


class ArmSim:
    """An interface for the three-link MapleSim model

    An interface for the three-link MapleSim model that has been exported
    to C and turned into shared libraries using Cython.

    Parameters
    ----------
    robot_config : class instance
        contains all relevant information about the arm
        such as: number of joints, number of links, mass information etc.
    dt: float, optional (Default: 0.001)
        simulation time step [seconds]
    q_init : numpy.array, optional (Default: robot_config.START_ANGLES)
        start joint angles [radians]
    dq_init : numpy.array, optional (Default: np.zeros)
        start joint velocity [radians/second]
    """

    def __init__(self, robot_config, dt=0.001, q_init=None, dq_init=None):

        self.robot_config = robot_config

        # create placeholders for joint angles and velocity
        self.q = np.zeros(self.robot_config.N_JOINTS)
        self.dq = np.zeros(self.robot_config.N_JOINTS)

        self.init_state = np.zeros(self.robot_config.N_JOINTS * 2)
        if q_init is None:
            self.init_state[::2] = self.robot_config.START_ANGLES
        else:
            self.init_state[::2] = q_init
        if dq_init is not None:
            self.init_state[1::2] = dq_init

        self.dt = dt  # time step

        self.torque_limit = 1e7  # max amplitude of torque signal allowed
        self.connect()

    def connect(self):
        """Creates the MapleSim model and set up PyGame."""

        # stores information returned from maplesim
        self.state = np.zeros(7)
        self.sim = pySim(dt=1e-5)

        self.reset()
        print("Connected to MapleSim model")

    def disconnect(self):
        """Reset the simulation and close PyGame display."""

        self.reset()
        print("MapleSim connection closed...")

    def reset(self):
        """Resets the state of the arm to starting conditions."""

        self.sim.reset(self.state, self.init_state)
        self._update_state()

    def send_forces(self, u, dt=None):
        """Apply the specified forces to the robot,
        moving the simulation one time step forward.

        NOTE: For this simulation, torques are clipped to 1e7
        to prevent seg faults being thrown.

        Parameters
        ----------
        u : numpy.array
            an array of the torques to apply to the robot [Nm]
        dt : float, optional (Default: self.dt)
            time step [seconds]
        """

        dt = self.dt if dt is None else dt
        # clip the torque signal to prevent seg faults
        u = np.minimum(
            np.maximum(-1 * np.array(u, dtype="float"), -self.torque_limit),
            self.torque_limit,
        )

        for _ in range(int(np.ceil(dt / 1e-5))):
            self.sim.step(self.state, u)
        self._update_state()

    def get_feedback(self):
        """Return a dictionary of information needed by the controller."""

        return {"q": self.q, "dq": self.dq}

    def get_xyz(self, name):
        """Not available in the MapleSim Interface"""

        raise NotImplementedError(
            "Not an available method" + "in the MapleSim interface"
        )

    def _position(self):
        """Compute x,y position of the hand"""

        xy = [
            self.robot_config.Tx(f"joint{ii}", q=self.q)
            for ii in range(self.robot_config.N_JOINTS)
        ]
        xy = np.vstack([xy, self.robot_config.Tx("EE", q=self.q)])
        self.joints_x = xy[:, 0]
        self.joints_y = xy[:, 1]
        return np.array([self.joints_x, self.joints_y])

    def _update_state(self):
        """Update the local variables"""

        self.t = self.state[0]
        self.q = self.state[1:4]
        self.dq = self.state[4:]
        self._position()
        self.x = np.array([self.joints_x[-1], self.joints_y[-1]])
