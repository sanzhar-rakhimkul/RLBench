from pyquaternion import Quaternion
import rlbench.gym
from gym import core, spaces
import gym
import numpy as np
from rlbench.observation_config import ObservationConfig, CameraConfig
from pyrep.const import RenderMode


ENV_SUPPORT = {'reach_target-state-v0': 100, 'reach_target-vision-v0':100,
               'reach_target_simple-state-v0': 100, 'reach_target_simple-vision-v0':100,
               'push_button-state-v0': 100, 'push_button-vision-v0': 100}
CAMERA_SUPPORT = ['left_shoulder_rgb', 'right_shoulder_rgb', 'wrist_rgb', 'front_rgb']
RENDER_MODE = RenderMode.OPENGL

# For action is joint space
class RLBenchWrapper(core.Env):
    def __init__(self,
                 env_name,
                 render=False,
                 action_scale=0.01):
        assert env_name in ENV_SUPPORT.keys(), 'Environment {} is not supported'.format(env_name)

        self._env_name = env_name
        self._render = render
        self._action_scale = action_scale

        # Create task
        self._make_env()

        # Create observation space
        # self._observation_space = self.env.observation_space
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10, ),
            dtype=np.float32
        )

        # Create action space
        self._action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_space.shape[0] - 1,),
            dtype=np.float32
        )


    def _make_env(self):
        render_mode = 'human' if self._render else None
        self.env = gym.make(self._env_name, render_mode=render_mode)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        obs_full = self.env.reset()
        return self._get_obs(obs_full)

    def step(self, action):
        action *= self._action_scale
        gripper = np.array([1])
        action = np.concatenate((action, gripper))

        obs, rew, done, info = self.env.step(action)

        success, _ = self.env.task._task.success()
        info.update(dict(success=success))

        reward = self.env.task._task.reward()
        rew = reward + float(success)
        # rew = reward
        return self._get_obs(obs), rew, done, info

    def compute_reward(self, act, obs, info=None):
        pass

    def _get_obs(self, obs_full):
        obs = np.concatenate((obs_full[7:14], obs_full[-3:]))
        return obs.copy()


class RLBenchWrapper_v1(core.Env):
    """
    This class supports to tool position mode only
    """
    def __init__(self,
                 env_name,
                 seed=1,
                 render=False,
                 action_scale=0.05,
                 height=84,
                 width=84,
                 frame_skip=1,
                 channels_first=True,
                 pixel_normalize=False,
                 z_offset=0.05,
                 xy_tolerance=0.1,
                 use_gripper=False,
                 allow_finish_soon=False,
                 use_rgb=True,
                 use_depth=False):
        """

        :param env_name: Name of environment, must be in ENV_SUPPORT list
        :param seed: Set seed for environment
        :param render: Rendering or not
        :param action_scale: Actual action = action from agent x action_scale
        :param height: of image
        :param width: of image
        :param frame_skip: An action will be repeated frame_skip time(s). This will not change the
        horizon of task, but change the horizon of VRESP env in simulator, e.g. if skip_frame = 1,
        the actual horizon in VRESP env is _max_episode_steps horizon, if skip_frame = 1, the actual
        horizon in VRESP env is 2*_max_episode_steps horizon, and so on.
        :param channels_first:
        :param pixel_normalize:
        :param z_offset: The gripper is always higher than table z_offset
        :param xy_tolerance: Make sure the gripper can reach to boundary.
        :param use_gripper:
        :param allow_finish_soon: Allowing terminate task when reach successful condition. It makes
        task's horizon varying.
        """
        assert env_name in ENV_SUPPORT.keys(), 'Environment {} is not supported'.format(env_name)

        self._env_name = env_name
        self._render = render
        self._action_scale = action_scale
        self._height = height
        self._width = width
        self._channels_first = channels_first
        self._pixel_normalize = pixel_normalize
        self._frame_skip = frame_skip
        self._reset_tool_pos = np.array([0.25, 0., 1.]) # Make sure gripper inside of boundary
        self._use_gripper = use_gripper
        self.z_offset = z_offset    # Make sure gripper always higher than table z_offset
        self.xy_tolerance = xy_tolerance
        self._allow_finish_soon = allow_finish_soon
        self.force_orientation = True
        self.tool_target_quat = None
        np.random.seed(seed)

        if 'state' in env_name:
            self._from_pixel = False
        elif 'vision' in env_name:
            self._from_pixel = True
        else:
            raise NotImplementedError('Not support this environment: {}'.format(env_name))

        if use_depth or use_rgb:
            assert self._from_pixel is True, 'Not support rgb/depth for env: {}'.format(env_name)
        self._use_rgb = use_rgb
        self._use_depth = use_depth
        self.key_obs_rgb = 'front_rgb'
        self.key_obs_depth = 'front_depth'

        self._make_env()

        # Create observation space
        if self._from_pixel:
            if self._use_rgb and self._use_depth:
                shape = [4, height, width] if channels_first else [height, width, 4]
            elif self._use_rgb and not self._use_depth:
                shape = [3, height, width] if channels_first else [height, width, 3]
            elif not self._use_rgb and self._use_depth:
                shape = [1, height, width] if channels_first else [height, width, 1]
            else:
                raise ValueError("Please set use_rgb or use_depth, or both to True.")
            if pixel_normalize:
                obs_low, obs_high = 0., 1.
                obs_type = np.float32
            else:
                obs_low, obs_high = 0, 255
                obs_type = np.uint8
        else:
            if self._use_gripper:
                shape = [3 + 3 + 1]     # Current tool's position + target's position
            else:
                shape = [3 + 3]
            obs_low, obs_high = -np.inf, np.inf
            obs_type = np.float32
        self._observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=shape, dtype=obs_type
        )

        # Create action space
        shape = (self.env.action_space.shape[0] - 4,) if use_gripper else (self.env.action_space.shape[0] - 5,)
        self._action_space = spaces.Box(
            low=-1.0, high=1.0, shape=shape, dtype=np.float32
        )

        # Get configurations of boundary
        self.ws_min, self.ws_max = self.get_range_boundary()

        self.step_cnt = None

    def _make_env(self):
        # Configure Camera
        use_left, use_right, use_wrist, use_front = False, False, False, True
        left_camera = CameraConfig(image_size=(self._height, self._width), render_mode=RENDER_MODE,
                                   rgb=use_left, depth=use_left, mask=use_left)
        right_camera = CameraConfig(image_size=(self._height, self._width), render_mode=RENDER_MODE,
                                    rgb=use_right, depth=use_right, mask=use_right)
        wrist_camera = CameraConfig(image_size=(self._height, self._width), render_mode=RENDER_MODE,
                                    rgb=use_wrist, depth=use_wrist, mask=use_wrist)
        front_camera = CameraConfig(image_size=(self._height, self._width), render_mode=RENDER_MODE,
                                    rgb=use_front, depth=use_front, mask=use_front)
        obs_config = ObservationConfig(left_shoulder_camera=left_camera,
                                       right_shoulder_camera=right_camera,
                                       wrist_camera=wrist_camera,
                                       front_camera=front_camera)
        self._max_episode_steps = ENV_SUPPORT[self._env_name]

        render_mode = 'human' if self._render else None
        self.env = gym.make(self._env_name, render_mode=render_mode, obs_config=obs_config)


    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def _current_tool_position(self):
        # Return position of tip of gripper
        return self.env.task._task.robot.arm.get_tip().get_position().copy()

    @property
    def _current_tool_quaternion(self):
        # Return quaternion of tip of gripper: (x, y, z, w)
        return self.env.env._robot.arm.get_tip().get_quaternion().copy()

    @property
    def _robot_state_low_dim(self):
        # Structure of get_low_dim_data for:
        # Reaching:
        #   gripper_open (1), joint_velocities (7), joint_positions (7), joint_forces (7),
        #   gripper_pose (7), gripper_joint_positions (2), gripper_touch_forces (6),
        #   task_low_dim_state (3)
        # Push Button:
        #   gripper_open (1), joint_velocities (7), joint_positions (7), joint_forces (7),
        #   gripper_pose (7), gripper_joint_positions (2), gripper_touch_forces (6),
        #   task_low_dim_state (43)
        return self.env.task._scene.get_observation().get_low_dim_data().copy()

    def reset(self):
        self.env.reset()
        # Keep tool's orientation same with reset's orientation
        self.tool_target_quat = self._current_tool_quaternion
        self.step_cnt = 0
        # This loop for move gripper to initial position
        while True:
            del_pos = self._reset_tool_pos - self._current_tool_position
            act = self._make_full_action(del_pos, 0.05, False)
            _, _, done, _ = self.env.step(act)
            if done:
                self.env.reset()
                continue
            if np.linalg.norm(del_pos) < 1e-3 and not done:
                break

        return self._get_obs()

    def step(self, act):
        pos = act[:3]
        gripper = act[-1] if self._use_gripper else None
        action = self._make_full_action(pos, gripper, self._action_scale)

        obs, done, info = None, False, {}
        for _ in range(self._frame_skip):
            if self._allow_finish_soon:
                obs, _, done, info = self.env.step(action)
                if done:
                    break   # Break if done from env, it means when success condition happened
            else:
                obs, _, _, info = self.env.step(action)

        # Control done flag by wrapper
        self.step_cnt += 1
        if self.step_cnt == self._max_episode_steps:
            done = True

        success, _ = self.env.task._task.success()
        info.update(dict(success=success))

        # TODO: Check that should cummulate reward across frame_skip?
        rew = self.compute_reward(action, obs)

        return self._get_obs(), rew, done, info

    def seed(self, seed=None):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def compute_reward(self, act, obs, info=None):
        if self.env.task._task.__class__.__name__ in ['ReachTarget', 'ReachTargetSimple']:
            success, _ = self.env.task._task.success()
            reward = -np.linalg.norm(self.env.task._task.target.get_position() -
                                     self.env.task._task.robot.arm.get_tip().get_position(), ord=2)
            # final_rew = 0.0 if success else -1.0
            total_rew = reward
        elif self.env.task._task.__class__.__name__ == 'PushButton':
            reward = -np.linalg.norm(self.env.task._task.target_button.get_position() -
                                     self.env.task._task.robot.arm.get_tip().get_position(), ord=2)
            total_rew = reward
        else:
            raise NotImplementedError('Not support compute reward for environment.')

        return total_rew

    def _compute_delta_orientation(self, target_orientation):
        """
            Compute the delta of quaternion of target's orientation relative with current one
        """
        qx, qy, qz, qw = self._current_tool_quaternion
        current_quat = Quaternion(qw, qx, qy, qz)
        tqx, tqy, tqz, tqw = target_orientation
        target_quat = Quaternion(tqw, tqx, tqy, tqz)
        delta_quat = target_quat/current_quat
        dqw, dqx, dqy, dqz = delta_quat
        return [dqx, dqy, dqz, dqw]

    def get_range_boundary(self):
        """
            Compute the x-, y-, z-range of workspace, add some tolerances to make sure the gripper
            can reach to boundary.
        """
        if hasattr(self.env.task._task, 'boundaries'):
            _boundary_center = self.env.task._task.boundaries.get_position()
            b_min_x, b_max_x, b_min_y, b_max_y, b_min_z, b_max_z = self.env.task._task.boundaries.get_bounding_box()
            b_min_x, b_max_x = (b_min_x - self.xy_tolerance, b_max_x + self.xy_tolerance)
            b_min_y, b_max_y = (b_min_y - self.xy_tolerance, b_max_y + self.xy_tolerance)
            b_min_z += self.z_offset  # Add this offset to void gripper collide with table
            ws_min = np.round(_boundary_center + np.array([b_min_x, b_min_y, b_min_z]), 2)
            ws_max = np.round(_boundary_center + np.array([b_max_x, b_max_y, b_max_z]), 2)
        else:
            # In tasks that don't provide boundary, using default boundary cloning from reach_target
            ws_min = np.array([-0.1 , -0.45,  0.75])    #  + self.z_offset
            ws_max = np.array([0.6 , 0.45, 1.25])

        return ws_min, ws_max

    def _make_full_action(self, pos, gripper=None, scale=1.0, check_valid=True):
        pos *= scale
        if check_valid:
            pos = self._check_workspace_valid(pos)

        if self._use_gripper:
            # The space of wrapper's gripper in [-1, 1] -> need to scale in range [0, 1]
            if gripper is not None:
                gripper = np.array([(gripper + 1.0) / 2])   # [0, 0.5): close, [0.5, 1]: open
            else:
                gripper = np.array([0])     # Close
        else:
            gripper = np.array([0])     # Close
        if self.force_orientation:
            quat = self._compute_delta_orientation(self.tool_target_quat)
            quat = np.array(quat)
        else:
            quat = np.array([0, 0, 0, 1])
        action_full = np.concatenate((pos, quat, gripper))
        return action_full

    def _get_obs(self):
        if self._from_pixel:
            # Get visual state after move to initial position
            obs_dict = self.env.task._scene.get_observation()
            if self._use_rgb and self._use_depth:
                # RGB-D
                obs_rgb = obs_dict.__dict__[self.key_obs_rgb]
                obs_depth = obs_dict.__dict__[self.key_obs_depth]
                obs = np.concatenate((obs_rgb, obs_depth[:, :, None]), axis=2)
            elif not self._use_rgb and self._use_depth:
                # D only
                obs_depth = obs_dict.__dict__[self.key_obs_depth]
                obs = obs_depth[:, :, None]
            elif self._use_rgb and not self._use_depth:
                # RGB
                obs_rgb = obs_dict.__dict__[self.key_obs_rgb]
                obs = obs_rgb
            else:
                raise NotImplementedError

            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
            if not self._pixel_normalize:
                obs = (obs * 255).astype(np.uint8).copy()
        else:
            obs_full = self._robot_state_low_dim
            cur_tool_pos = obs_full[22:25].copy()
            if self.env.task._task.__class__.__name__ in ['ReachTarget', 'ReachTargetSimple']:
                target_pos = obs_full[-3:].copy()
                obs = np.concatenate((cur_tool_pos, target_pos))
                if self._use_gripper:
                    gripper_state = obs_full[0, None]
                    obs = np.concatenate((obs, gripper_state))
            elif self.env.task._task.__class__.__name__ == 'PushButton':
                button_pos = obs_full[37:40]
                obs = np.concatenate((cur_tool_pos, button_pos))
                if self._use_gripper:
                    gripper_state = obs_full[0, None]
                    obs = np.concatenate((obs, gripper_state))
            else:
                raise NotImplementedError('Not support compute reward for environment.')
        return obs.copy()

    def _check_workspace_valid(self, action):
        # Note: action is delta relative to current position
        # Get expected next position
        tool_next_position = self._current_tool_position + action
        # Make sure next action is valid, i.e.: within workspace (boundaries)
        tool_next_position = np.clip(tool_next_position, self.ws_min, self.ws_max)
        # Get action (delta) after truncateing
        truncated_action = tool_next_position - self._current_tool_position
        # if not np.allclose(action, truncated_action, atol=1e-6):
        #     print('[DEBUG]: Truncating action')
        return truncated_action


if __name__ == '__main__':
    # env = RLBenchWrapper(env_name='reach_target-state-v0', render=True)
    env = RLBenchWrapper_v1(env_name='reach_target-state-v0', render=True)
    obs = env.reset()
    for i in range(100):
        print('step: ', i)
        obs, rew, done, info = env.step(env.action_space.sample())

