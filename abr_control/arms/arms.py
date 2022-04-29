#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 19:23:15 2022

@author: raviswarooprayavarapu
"""

import numpy as np
#from abr_control.interfaces.mujoco import Mujoco



class arms:
    
    def __init__(self,robot_config,sim,joint_names,joint_ids):
        
        self.robot_config = robot_config
        self.sim = sim
        model = self.sim.model
        
        joint_addr = model.body_jntadr
        bodyid_ee = np.where(joint_addr == joint_ids[-1])[0]
        while bodyid_ee < len(joint_addr)-1 and joint_addr[bodyid_ee+1] == -1:
            bodyid_ee += 1
        bodyid_ee = bodyid_ee[0]
        self.ee = model.body_id2name(bodyid_ee)
        
        self.joint_pos_addrs = [model.get_joint_qpos_addr(name) for name in joint_names]
        self.joint_vel_addrs = [model.get_joint_qvel_addr(name) for name in joint_names]
        
        joint_pos_addrs = []
        for elem in self.joint_pos_addrs:
            if isinstance(elem, tuple):
                joint_pos_addrs += list(range(elem[0], elem[1]))
            else:
                joint_pos_addrs.append(elem)
        self.joint_pos_addrs = joint_pos_addrs

        joint_vel_addrs = []
        for elem in self.joint_vel_addrs:
            if isinstance(elem, tuple):
                joint_vel_addrs += list(range(elem[0], elem[1]))
            else:
                joint_vel_addrs.append(elem)
        self.joint_vel_addrs = joint_vel_addrs

        # Need to also get the joint rows of the Jacobian, inertia matrix, and
        # gravity vector. This is trickier because if there's a quaternion in
        # the joint (e.g. a free joint or a ball joint) then the joint position
        # address will be different than the joint Jacobian row. This is because
        # the quaternion joint will have a 4D position and a 3D derivative. So
        # we go through all the joints, and find out what type they are, then
        # calculate the Jacobian position based on their order and type.
        index = self.joint_pos_addrs[0]
        self.joint_dyn_addrs = []
        for ii, joint_type in enumerate(model.jnt_type):
            if ii in joint_ids:
                self.joint_dyn_addrs.append(index)
                if joint_type == 0:  # free joint
                    self.joint_dyn_addrs += [jj + index for jj in range(1, 6)]
                    index += 6  # derivative has 6 dimensions
                elif joint_type == 1:  # ball joint
                    self.joint_dyn_addrs += [jj + index for jj in range(1, 3)]
                    index += 3  # derivative has 3 dimension
                else:  # slide or hinge joint
                    index += 1  # derivative has 1 dimensions

        # give the robot config access to the sim for wrapping the
        # forward kinematics / dynamics functions
        