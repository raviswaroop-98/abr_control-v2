"""
Move the jaco2 to a target position with an adaptive controller
that will account for a 2lb weight in its hand
"""
import numpy as np
import os
import timeit
import traceback
import redis

from abr_control.controllers import OSC, signals, path_planners
import abr_jaco2
import nengo

class Training:

    def __init__(self):
        pass

    def run_test(self, n_neurons=1000, n_ensembles=1, decimal_scale=1,
                 test_name="adaptive_training", session=None, run=None,
                 weights_file=None ,pes_learning_rate=1e-6, backend=None,
                 autoload=False, time_limit=30, vision_target=False,
                 offset=None, avoid_limits=False, additional_mass=0,
                 kp=20, kv=6, ki=0, multi_target=False):
        #TODO: Add name of paper once complete
        #TODO: Do we want to include nengo_spinnaker install instructions?
        #TODO: Add option to use our saved results incase user doesn't have
        # spinnaker
        """
        The test script used to collect data for training. The use of the
        adaptive controller and the type of backend can be specified. The script is
        also automated to use the correct weights files for continuous learning.
        The standard naming convention used is runs are consecutive tests where the previous
        learned weights are used. The whole of these runs are a single session.
        Multiple sessions of these runs can then be used for averaging and
        statictical purposes.

        The adaptive controller uses a 6 dimensional input and returns a 3 dimensional
        output. The base, shoulder and elbow joints' positions and velocities are
        used as the input, and the control signal for the corresponding joints gets
        returned.

        Parameters
        ----------
        n_neurons: int, Optional (Default: 1000)
            the number of neurons in the adaptive population
        n_ensembles: int, Optional (Default: 1)
            the number of ensembles of n_neurons number of neurons
        decimal_scale: int, Optional (Default: 1)
            used for scaling the spinnaker input. Due to precision limit, spinnaker
            input gets scaled to allow for more decimals, then scaled back once
            extracted. Also can be used to account for learning speed since
            spinnaker runs in real time
        test_name: string, Optional (Default: "dewolf_2017_data")
            folder name where data is saved
        session: int, Optional (Default: None)
            The current session number, if left to None then it will be automatically
            updated based on the current session. If the next session is desired then
            this will need to be updated.
        run: int, Optional (Default: None)
            The current nth run that specifies to use the weights from the n-1 run.
            If set to None if will automatically increment based off the last saved
            weights.
        weights_file: string, Optional (Default: None)
            the path to the desired saved weights to use. If None will
            automatically take the most recent weights saved in the 'test_name'
            directory
        pes_learning_rate: float, Optional (Default: 1e-6)
            the learning rate for the adaptive population
        backend: string
            None: non adaptive control, Optional (Default: None)
            'nengo': use nengo as the backend for the adaptive population
            'spinnaker': use spinnaker as the adaptive population
        autoload: boolean, Optional (Default: False)
            True: used the specified weights, or the last set of learned weights if
                  not specified
            False: use the specified weights, if not specified start learning from
                   zero
        time_limit: float, Optional (Default: 30)
            the time limit for each run in seconds
        vision_target: boolean, Optional (Default: False)
            True: check redis server for targets sent from vision
            False: use hardcoded target locations
        offset: float array, Optional (Default: None)
            Set the offset to the end effector if something other than the default
            is desired. Use the form [x_offset, y_offset, z_offset]
        avoid_limits: boolean, Optional (Default: False)
            set true if there are limits you would like to avoid
        additional_mass: float, Optional (Default: 0)
            any extra mass added to the EE if known [kg]
        kp: float, Optional (Default: 20)
            proportional gain term
        kv: float, Optional (Default: 6)
            derivative gain term
        ki: float, Optional (Default: 0)
            integral gain term
        multi_target: boolean, Optional (Default: False)
            False: single target
            True: multiple targets
        """

        self.decimal_scale = decimal_scale

        # try to setup redis server if vision targets are desired
        if vision_target:
            try:
                import redis
                redis_server = redis.StrictRedis(host='localhost')
                self.redis_server = redis_server
            except ImportError:
                print('ERROR: Install redis to use vision targets, using preset targets')
                vision_target = False

        if multi_target:
            PRESET_TARGET = np.array([[.56, -.09, .95],
                                      [.62, .15, .80],
                                      [.59, .35, .61],
                                      [.38, .55, .81],
                                      [.10, .51, .95]])
            time_limit /= len(PRESET_TARGET)
        else:
            PRESET_TARGET = np.array([[.57, 0.03, .87]])

        # initialize our robot config
        robot_config = abr_jaco2.Config(
            use_cython=True, hand_attached=True)
        self.robot_config = robot_config

        zeros = np.zeros(robot_config.N_JOINTS)

        # get Jacobians to each link for calculating perturbation
        self.J_links = [robot_config._calc_J('link%s' % ii, x=[0, 0, 0])
                   for ii in range(robot_config.N_LINKS)]

        self.JEE  = robot_config._calc_J('EE', x=[0, 0, 0])

        # account for wrist to fingers offset
        R_func = robot_config._calc_R('EE')

        # Use user defined offset if one is specified
        if offset is None:
            OFFSET = robot_config.OFFSET
        else:
            OFFSET = offset

        robot_config.Tx('EE', q=zeros, x=OFFSET)

        # temporarily set backend to nengo if non adaptive if selected so we can
        # still use the weight saving function in dynamics_adaptation
        if backend is None:
            adapt_backend = 'nengo'
        else:
            adapt_backend = backend
        # create our adaptive controller
        n_input = 4
        n_output = 2
        # number of joint angle dim in
        self.adapt_dim = 2

        self.additional_mass = additional_mass
        if self.additional_mass != 0 and weights_file is None:
            # if the user knows about the mass at the EE, try and improve
            # our starting point
            # starting estimate of mass if using function
            self.fake_gravity = np.array([[0, 0, -9.81*self.additional_mass, 0, 0, 0]]).T
            print('Using mass estimate of %f kg as starting point'
                  %self.additional_mass)
            adapt = signals.DynamicsAdaptation(
                n_input=n_input,
                n_output=n_output,
                n_neurons=n_neurons,
                n_ensembles=n_ensembles,
                pes_learning_rate=pes_learning_rate,
                intercepts=(-0.1, 1.0),
                weights_file=weights_file,
                backend=adapt_backend,
                session=session,
                run=run,
                test_name=test_name,
                autoload=autoload,
                function=self.gravity_estimate)

        else:
            adapt = signals.DynamicsAdaptation(
                n_input=n_input,
                n_output=n_output,
                n_neurons=n_neurons,
                n_ensembles=n_ensembles,
                pes_learning_rate=pes_learning_rate,
                intercepts=(-0.1, 1.0),
                weights_file=weights_file,
                backend=adapt_backend,
                session=session,
                run=run,
                test_name=test_name,
                autoload=autoload)

        # get save location of weights to save tracked data in same directory
        [location, run_num] = adapt.weights_location(test_name=test_name, run=run,
                                                     session=session)
        if ki != 0:
            if run_num < 1:
                run_num = 0
                int_err_prev = [0,0,0]
            else:
                int_err_prev = np.squeeze(np.load(location + '/run%i_data/int_err%i.npz'
                                       % (run_num-1, run_num-1))['int_err'])[-1]
        else:
            int_err_prev = [0, 0, 0]

        # instantiate controller and path planner
        ctrlr = OSC(robot_config, kp=kp, kv=kv, ki=ki, vmax=1,
                    null_control=True, int_err=int_err_prev)
        if avoid_limits:
            avoid = signals.AvoidJointLimits(
                      robot_config,
                      # joint 4 flipped because does not cross 0-2pi line
                      min_joint_angles=[0.8, 1.1, 0.5, 3.5, 2.0, 1.6],
                      max_joint_angles=[4.75, 3.65, 6.25, 6.0, 5.0, 4.6],
                      max_torque=[5]*robot_config.N_JOINTS,
                      cross_zero=[True, False, False, False, True, False],
                      gradient = [False, True, True, True, True, False])
        path = path_planners.SecondOrder(robot_config)
        n_timesteps = 4000
        w = 1e4/n_timesteps
        zeta = 2
        dt = 0.003

        # run controller once to generate functions / take care of overhead
        # outside of the main loop, because force mode auto-exits after 200ms
        ctrlr.generate(zeros, zeros, np.zeros(3), offset=OFFSET)

        interface = abr_jaco2.Interface(robot_config)



        # connect to and initialize the arm
        interface.connect()
        interface.init_position_mode()
        interface.send_target_angles(robot_config.INIT_TORQUE_POSITION)

        # set up lists for tracking data
        time_track = []
        q_track = []
        u_track = []
        adapt_track = []
        error_track = []
        training_track = []
        target_track = []
        ee_track = []
        #input_signal = []
        if ki != 0:
            int_err_track = []
        try:
            interface.init_force_mode()
            for ii in range(0,len(PRESET_TARGET)):
                # get the end-effector's initial position
                feedback = interface.get_feedback()

                # counter for print statements
                count = 0

                # track loop_time for stopping test
                loop_time = 0

                # get joint angle and velocity feedback
                feedback = interface.get_feedback()
                q = feedback['q']
                dq = feedback['dq']

                # calculate end-effector position
                ee_xyz = robot_config.Tx('EE', q=q, x= OFFSET)

                # last three terms used as started point for target velocity of
                # base 3 joints
                target = np.concatenate((ee_xyz, np.array([0, 0, 0])), axis=0)

                while loop_time < time_limit:
                    if vision_target is False:
                        TARGET_XYZ = PRESET_TARGET[ii]
                    else:
                        TARGET_XYZ = self.get_target_from_camera()
                        TARGET_XYZ = self.normalize_target(TARGET_XYZ)

                    start = timeit.default_timer()
                    prev_xyz = ee_xyz
                    target = path.step(y=target[:3], dy=target[3:], target=TARGET_XYZ, w=w,
                                       zeta=zeta, dt=dt, threshold=0.05)

                    # get joint angle and velocity feedback
                    feedback = interface.get_feedback()
                    q = feedback['q']
                    dq = feedback['dq']

                    # calculate end-effector position
                    ee_xyz = robot_config.Tx('EE', q=q, x= OFFSET)

                    # calculate the control signal
                    u_base = ctrlr.generate(
                        q=q,
                        dq=dq ,
                        target_pos=target[:3],
                        target_vel=target[3:],
                        offset = OFFSET)

                    # account for stiction in jaco2 base
                    if u_base[0] > 0:
                        u_base[0] *= 3.0
                    else:
                        u_base[0] *= 3.0 #2.0
                    training_signal = np.array([ctrlr.training_signal[1],
                                                ctrlr.training_signal[2]])
                                                # ctrlr.training_signal[4]])

                    if backend is not None:
                        # calculate the adaptive control signal
                        adapt_input = np.array([robot_config.scaledown('q',q)[1],
                                                   robot_config.scaledown('q',q)[2],
                                                   # robot_config.scaledown('q',q)[4],
                                                   robot_config.scaledown('dq',dq)[1],
                                                   robot_config.scaledown('dq',dq)[2]])
                        u_adapt = adapt.generate(input_signal=adapt_input,
                                                 training_signal=training_signal)
                    else:
                        u_adapt = np.zeros(6)

                    # add adaptive signal to base controller
                    u_adapt = np.array([0,
                                        u_adapt[0]/decimal_scale,
                                        u_adapt[1]/decimal_scale,
                                        0,
                                        0,
                                        #u_adapt[2]/decimal_scale,
                                        0])
                    u = u_base + u_adapt

                    # add limit avoidance if True
                    if avoid_limits:
                        u += avoid.generate(q)

                    # send forces
                    interface.send_forces(np.array(u, dtype='float32'))

                    # calculate euclidean distance to target
                    error = np.sqrt(np.sum((ee_xyz - TARGET_XYZ)**2))

                    # track data
                    q_track.append(np.copy(q))
                    u_track.append(np.copy(u))
                    adapt_track.append(np.copy(u_adapt))
                    error_track.append(np.copy(error))
                    training_track.append(np.copy(training_signal))
                    end = timeit.default_timer() - start
                    loop_time += end
                    time_track.append(np.copy(end))
                    target_track.append(np.copy(TARGET_XYZ))
                    #input_signal.append(np.copy(adapt_input))
                    ee_track.append(np.copy(ee_xyz))
                    if ki != 0:
                        int_err_track.append(np.copy(ctrlr.int_err))

                    if count % 1000 == 0:
                        print('error: ', error)
                        print('dt: ', end)
                        print('adapt: ', u_adapt)
                        print('int_err: ', ctrlr.int_err*ki)
                        #print('q: ', q)
                        #print('hand: ', ee_xyz)
                        #print('target: ', target)
                        #print('control: ', u_base)

                    count += 1

        except:
            print(traceback.format_exc())

        finally:
            # close the connection to the arm
            interface.init_position_mode()

            if backend != None:
                # Save the learned weights
                adapt.save_weights(test_name=test_name, session=session, run=run)

            interface.send_target_angles(robot_config.INIT_TORQUE_POSITION)
            interface.disconnect()

            print('Average loop speed: ', sum(time_track)/len(time_track))
            print('Run number ', run_num)
            print('Saving tracked data to ', location + '/run%i_data' % (run_num))

            # get save location of weights to save tracked data in same directory
            [location, run_num] = adapt.weights_location(test_name=test_name, run=run,
                                                         session=session)

            time_track = np.array(time_track)
            q_track = np.array(q_track)
            u_track = np.array(u_track)
            adapt_track = np.array(adapt_track)
            error_track = np.array(error_track)
            training_track = np.array(training_track)
            #input_signal = np.array(input_signal)
            ee_track = np.array(ee_track)
            if ki != 0:
                int_err_track = np.array(int_err_track)

            if not os.path.exists(location + '/run%i_data' % (run_num)):
                os.makedirs(location + '/run%i_data' % (run_num))

            np.savez_compressed(location + '/run%i_data/q%i' % (run_num, run_num),
                                q=[q_track])
            np.savez_compressed(location + '/run%i_data/time%i' % (run_num, run_num),
                                time=[time_track])
            np.savez_compressed(location + '/run%i_data/u%i' % (run_num, run_num),
                                u=[u_track])
            np.savez_compressed(location + '/run%i_data/adapt%i' % (run_num, run_num),
                                adapt=[adapt_track])
            np.savez_compressed(location + '/run%i_data/target%i' % (run_num, run_num),
                                target=[target_track])
            np.savez_compressed(location + '/run%i_data/error%i' % (run_num, run_num),
                                error=[error_track])
            np.savez_compressed(location + '/run%i_data/training%i' % (run_num, run_num),
                                training=[training_track])
            # np.savez_compressed(location + '/run%i_data/input_signal%i' % (run_num, run_num),
            #                     input_signal=[input_signal])
            np.savez_compressed(location + '/run%i_data/ee_xyz%i' % (run_num, run_num),
                                ee_xyz=[ee_track])
            if ki != 0:
                np.savez_compressed(location + '/run%i_data/int_err%i' % (run_num, run_num),
                                    int_err=[int_err_track])
    def get_target_from_camera(self):
        # read from server
        camera_xyz = self.redis_server.get('target_xyz').decode('ascii')
        # if the target has changed, recalculate things
        camera_xyz = np.array([float(val) for val in camera_xyz.split()])
        # transform from camera to robot reference frame
        target_xyz = self.robot_config.Tx(
            'camera', x=camera_xyz, q=np.zeros(6))

        self.redis_server.set(
            'target_xyz_robot_coords', '%.3f %.3f %.3f' % tuple(target_xyz))

        return target_xyz

    def normalize_target(self, target, magnitude=0.9):
        # set it so that target is not too far from joint 1
        joint1_offset = np.array([0, 0, 0.273])
        norm = np.linalg.norm(target - joint1_offset)
        if norm > magnitude:
            target = ((target - joint1_offset) / norm) * magnitude + joint1_offset
        return target

    # def gravity_estimate(self, x):
    #     fake_gravity = np.array([[0, -9.81*self.additional_mass, 0, 0, 0, 0]]).T
    #     q = self.robot_config.scaleup('q', x[:2])
    #     g = np.zeros((self.robot_config.N_JOINTS, 1))
    #     for ii in range(self.robot_config.N_LINKS):
    #         pars = tuple(q) + tuple([0, 0, 0])
    #         g += np.dot(J_links[ii](*pars).T, fake_gravity)
    #     return -g[[1,2]]

    # def gravity_estimate(self, x):
    #     q1, q2 = x[:self.adapt_dim]  # x = [q, dq]
    #     g_avg = []
    #     dist = nengo.dists.UniformHypersphere()
    #     samples = dist.sample(1000, d=self.robot_config.N_JOINTS - self.adapt_dim)
    #     for sample in samples:
    #         q = np.hstack([sample[0], q1, q2, sample[1:]])
    #         q = self.robot_config.scaleup('q', q)
    #         g = np.zeros((self.robot_config.N_JOINTS, 1))
    #         for ii in range(self.robot_config.N_LINKS):
    #             pars = tuple(q) + tuple([0, 0, 0])
    #             g += np.dot(self.J_links[ii](*pars).T, self.fake_gravity)
    #         g_avg.append(g.squeeze())
    #     g_avg = np.mean(np.array(g_avg), axis=0)[[1, 2]]
    #     return -g_avg

    def gravity_estimate(self, x):
        q1, q2 = x[:self.adapt_dim]  # x = [q, dq]
        g_avg = []
        dist = nengo.dists.UniformHypersphere()
        samples = dist.sample(1000, d=self.robot_config.N_JOINTS - self.adapt_dim)
        for sample in samples:
            q = np.hstack([sample[0], q1, q2, sample[1:]])
            q = self.robot_config.scaleup('q', q)
            pars = tuple(q) + tuple([0, 0, 0])
            g = np.dot(self.JEE(*pars).T, self.fake_gravity)
            g_avg.append(g.squeeze())
        g_avg = np.mean(np.array(g_avg), axis=0)[[1, 2]]
        return -g_avg*self.decimal_scale