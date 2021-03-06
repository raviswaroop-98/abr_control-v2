# Config file for UR5 in VREP
import os

import numpy as np
import sympy as sp

from ..base_config import BaseConfig


class Config(BaseConfig):
    """Robot config file for the UR5

    Attributes
    ----------
    START_ANGLES : numpy.array
        The joint angles for a safe home or rest position
    _M_LINKS : sympy.diag
        inertia matrix of the links
    _M_JOINTS : sympy.diag
        inertia matrix of the joints
    L : numpy.array
        segment lengths of arm [meters]
    L_HANDCOM : numpy.array
        offset to the center of mass of the hand [meters]

    Transform Naming Convention: Tpoint1point2
    ex: Tj1l1 transforms from joint 1 to link 1

    Transforms are broken up into two matrices for simplification
    ex: Tj0l1a and Tj0l1b where the former transform accounts for
    joint rotations and the latter accounts for static rotations
    and translations
    """

    def __init__(self, **kwargs):

        super().__init__(N_JOINTS=6, N_LINKS=7, ROBOT_NAME="ur5", **kwargs)
        self.filename = f"{os.path.dirname(os.path.abspath(__file__))}/ur5.ttt"
        self.google_id = "1EDM6H9hbFhCjcsfm0p2lQ1K55o5Yi1VV"
        self._T = {}  # dictionary for storing calculated transforms

        self.JOINT_NAMES = [f"UR5_joint{ii}" for ii in range(self.N_JOINTS)]

        self.START_ANGLES = np.array(
            [0, np.pi / 4.0, -np.pi / 2.0, np.pi / 4.0, np.pi / 2.0, np.pi / 2.0],
            dtype="float32",
        )

        # TODO: automate getting all this information from VREP

        # create the inertia matrices for each link of the ur5
        self._M_LINKS = [
            sp.diag(1.0, 1.0, 1.0, 0.02, 0.02, 0.02),  # link0
            sp.diag(2.5, 2.5, 2.5, 0.04, 0.04, 0.04),  # link1
            sp.diag(5.7, 5.7, 5.7, 0.06, 0.06, 0.04),  # link2
            sp.diag(3.9, 3.9, 3.9, 0.055, 0.055, 0.04),  # link3
            sp.diag(2.5, 2.5, 2.5, 0.04, 0.04, 0.04),  # link4
            sp.diag(2.5, 2.5, 2.5, 0.04, 0.04, 0.04),  # link5
            sp.diag(0.7, 0.7, 0.7, 0.01, 0.01, 0.01),
        ]  # link6

        # the joints don't weigh anything in VREP
        self._M_JOINTS = [sp.zeros(6, 6) for ii in range(self.N_JOINTS)]

        # segment lengths associated with each transform
        # ignoring lengths < 1e-6
        self.L = np.array(
            [
                [0.0, 0.0, 1.4650e-02],  # link 0 offset
                [0.0, 0.0, 8.5001e-03],  # joint 0 offset
                [-7.1771e-03, 1.1159e-04, 7.0381e-02],  # link 1 offset
                [-6.3122e-02, -9.5099e-05, -4.3305e-03],  # joint 1 offset
                [2.1255e-01, -9.9446e-04, 6.4234e-02],  # link 2 offset
                [6.4235e-02, 1.1502e-04, 2.1255e-01],  # joint 2 offset
                [1.8677e-01, 6.7934e-04, -5.7847e-02],  # link 3 offset
                [-5.7847e-02, -1.6153e-05, 2.0538e-01],  # joint 3 offset
                [-7.5028e-03, -5.5328e-05, 3.2830e-02],  # link 4 offset
                [-6.8700e-03, 4.5318e-05, 5.3076e-02],  # joint 4 offset
                [3.6091e-03, 5.0090e-05, 4.2340e-02],  # link 5 offset
                [1.0824e-02, -4.5293e-05, 6.8700e-03],  # joint 5 offset
                [0, 0, 7.6645e-02],
            ]
        )  # link 6 offset

        # ---- Joint Transform Matrices ----

        # Transform matrix : origin -> link 0
        # no change of axes, account for offsets
        self.Torgl0 = sp.Matrix(
            [
                [1, 0, 0, self.L[0, 0]],
                [0, 1, 0, self.L[0, 1]],
                [0, 0, 1, self.L[0, 2]],
                [0, 0, 0, 1],
            ]
        )

        # Transform matrix : link 0 -> joint 0
        # no change of axes, account for offsets
        self.Tl0j0 = sp.Matrix(
            [
                [1, 0, 0, self.L[1, 0]],
                [0, 1, 0, self.L[1, 1]],
                [0, 0, 1, self.L[1, 2]],
                [0, 0, 0, 1],
            ]
        )

        # Transform matrix : joint 0 -> link 1
        # account for rotations due to q
        self.Tj0l1a = sp.Matrix(
            [
                [sp.cos(self.q[0]), -sp.sin(self.q[0]), 0, 0],
                [sp.sin(self.q[0]), sp.cos(self.q[0]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        # no change of axes, account for offsets
        self.Tj0l1b = sp.Matrix(
            [
                [1, 0, 0, self.L[2, 0]],
                [0, 1, 0, self.L[2, 1]],
                [0, 0, 1, self.L[2, 2]],
                [0, 0, 0, 1],
            ]
        )
        self.Tj0l1 = self.Tj0l1a * self.Tj0l1b

        # Transform matrix : link 1 -> joint 1
        # account for axes rotation and offset
        self.Tl1j1 = sp.Matrix(
            [
                [0, 0, -1, self.L[3, 0]],
                [0, 1, 0, self.L[3, 1]],
                [1, 0, 0, self.L[3, 2]],
                [0, 0, 0, 1],
            ]
        )

        # Transform matrix : joint 1 -> link 2
        # account for rotations due to q
        self.Tj1l2a = sp.Matrix(
            [
                [sp.cos(self.q[1]), -sp.sin(self.q[1]), 0, 0],
                [sp.sin(self.q[1]), sp.cos(self.q[1]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        # account for axes rotation and offsets
        self.Tj1l2b = sp.Matrix(
            [
                [0, 0, 1, self.L[4, 0]],
                [0, 1, 0, self.L[4, 1]],
                [-1, 0, 0, self.L[4, 2]],
                [0, 0, 0, 1],
            ]
        )
        self.Tj1l2 = self.Tj1l2a * self.Tj1l2b

        # Transform matrix : link 2 -> joint 2
        # account for axes rotation and offsets
        self.Tl2j2 = sp.Matrix(
            [
                [0, 0, -1, self.L[5, 0]],
                [0, 1, 0, self.L[5, 1]],
                [1, 0, 0, self.L[5, 2]],
                [0, 0, 0, 1],
            ]
        )

        # Transform matrix : joint 2 -> link 3
        # account for rotations due to q
        self.Tj2l3a = sp.Matrix(
            [
                [sp.cos(self.q[2]), -sp.sin(self.q[2]), 0, 0],
                [sp.sin(self.q[2]), sp.cos(self.q[2]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        # account for axes rotation and offsets
        self.Tj2l3b = sp.Matrix(
            [
                [0, 0, 1, self.L[6, 0]],
                [0, 1, 0, self.L[6, 1]],
                [-1, 0, 0, self.L[6, 2]],
                [0, 0, 0, 1],
            ]
        )
        self.Tj2l3 = self.Tj2l3a * self.Tj2l3b

        # Transform matrix : link 3 -> joint 3
        # account for axes change and offsets
        self.Tl3j3 = sp.Matrix(
            [
                [0, 0, -1, self.L[7, 0]],
                [0, 1, 0, self.L[7, 1]],
                [1, 0, 0, self.L[7, 2]],
                [0, 0, 0, 1],
            ]
        )

        # Transform matrix: joint 3 -> link 4
        # account for rotations due to q
        self.Tj3l4a = sp.Matrix(
            [
                [sp.cos(self.q[3]), -sp.sin(self.q[3]), 0, 0],
                [sp.sin(self.q[3]), sp.cos(self.q[3]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        # account for axes and rotation and offsets
        self.Tj3l4b = sp.Matrix(
            [
                [0, 0, 1, self.L[8, 0]],
                [0, 1, 0, self.L[8, 1]],
                [-1, 0, 0, self.L[8, 2]],
                [0, 0, 0, 1],
            ]
        )
        self.Tj3l4 = self.Tj3l4a * self.Tj3l4b

        # Transform matrix: link 4 -> joint 4
        # no axes change, account for offsets
        self.Tl4j4 = sp.Matrix(
            [
                [1, 0, 0, self.L[9, 0]],
                [0, 1, 0, self.L[9, 1]],
                [0, 0, 1, self.L[9, 2]],
                [0, 0, 0, 1],
            ]
        )

        # Transform matrix: joint 4 -> link 5
        # account for rotations due to q
        self.Tj4l5a = sp.Matrix(
            [
                [sp.cos(self.q[4]), -sp.sin(self.q[4]), 0, 0],
                [sp.sin(self.q[4]), sp.cos(self.q[4]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        # account for axes and rotation and offsets
        # no axes change, account for offsets
        self.Tj4l5b = sp.Matrix(
            [
                [1, 0, 0, self.L[10, 0]],
                [0, 1, 0, self.L[10, 1]],
                [0, 0, 1, self.L[10, 2]],
                [0, 0, 0, 1],
            ]
        )
        self.Tj4l5 = self.Tj4l5a * self.Tj4l5b

        # Transform matrix : link 5 -> joint 5
        # account for axes change and offsets
        self.Tl5j5 = sp.Matrix(
            [
                [0, 0, -1, self.L[11, 0]],
                [0, 1, 0, self.L[11, 1]],
                [1, 0, 0, self.L[11, 2]],
                [0, 0, 0, 1],
            ]
        )

        # Transform matrix: joint 5 -> link 6
        # account for rotations due to q
        self.Tj5l6a = sp.Matrix(
            [
                [sp.cos(self.q[5]), -sp.sin(self.q[5]), 0, 0],
                [sp.sin(self.q[5]), sp.cos(self.q[5]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        # no axes change, account for offsets
        self.Tj5l6b = sp.Matrix(
            [
                [1, 0, 0, self.L[12, 0]],
                [0, 1, 0, self.L[12, 1]],
                [0, 0, 1, self.L[12, 2]],
                [0, 0, 0, 1],
            ]
        )
        self.Tj5l6 = self.Tj5l6a * self.Tj5l6b

        # orientation part of the Jacobian (compensating for angular velocity)
        self.J_orientation = [
            self._calc_T("joint0")[:3, :3] * self._KZ,  # joint 0 orientation
            self._calc_T("joint1")[:3, :3] * self._KZ,  # joint 1 orientation
            self._calc_T("joint2")[:3, :3] * self._KZ,  # joint 2 orientation
            self._calc_T("joint3")[:3, :3] * self._KZ,  # joint 3 orientation
            self._calc_T("joint4")[:3, :3] * self._KZ,  # joint 4 orientation
            self._calc_T("joint5")[:3, :3] * self._KZ,
        ]  # joint 5 orientation

    def _calc_T(self, name):  # noqa C907
        """Uses Sympy to generate the transform for a joint or link

        name : string
            name of the joint, link, or end-effector
        """

        if self._T.get(name, None) is None:
            if name == "link0":
                self._T[name] = self.Torgl0
            elif name == "joint0":
                self._T[name] = self._calc_T("link0") * self.Tl0j0
            elif name == "link1":
                self._T[name] = self._calc_T("joint0") * self.Tj0l1
            elif name == "joint1":
                self._T[name] = self._calc_T("link1") * self.Tl1j1
            elif name == "link2":
                self._T[name] = self._calc_T("joint1") * self.Tj1l2
            elif name == "joint2":
                self._T[name] = self._calc_T("link2") * self.Tl2j2
            elif name == "link3":
                self._T[name] = self._calc_T("joint2") * self.Tj2l3
            elif name == "joint3":
                self._T[name] = self._calc_T("link3") * self.Tl3j3
            elif name == "link4":
                self._T[name] = self._calc_T("joint3") * self.Tj3l4
            elif name == "joint4":
                self._T[name] = self._calc_T("link4") * self.Tl4j4
            elif name == "link5":
                self._T[name] = self._calc_T("joint4") * self.Tj4l5
            elif name == "joint5":
                self._T[name] = self._calc_T("link5") * self.Tl5j5
            elif name in ("link6", "EE"):
                self._T[name] = self._calc_T("joint5") * self.Tj5l6

            else:
                raise Exception(f"Invalid transformation name: {name}")

        return self._T[name]
