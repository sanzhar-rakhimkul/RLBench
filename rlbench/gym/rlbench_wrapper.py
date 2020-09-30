from pyquaternion import Quaternion
import rlbench.gym
from gym import core, spaces
import gym
import numpy as np
from rlbench.observation_config import ObservationConfig, CameraConfig


ENV_SUPPORT = ['reach_target-state-v0', 'reach_target-vision-v0']
CAMERA_SUPPORT = ['left_shoulder_rgb', 'right_shoulder_rgb', 'wrist_rgb', 'front_rgb']


# For action is joint space
class RLBenchWrapper(core.Env):
    def __init__(self,
                 env_name,
                 render=False,
                 action_scale=0.01):
        assert env_name in ENV_SUPPORT, 'Environment {} is not supported'.format(env_name)

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
                 render=False,
                 action_scale=0.05,
                 height=84,
                 width=84,
                 z_offset=0.05,
                 xy_tolerance=0.1):
        assert env_name in ENV_SUPPORT, 'Environment {} is not supported'.format(env_name)

        self._env_name = env_name
        self._render = render
        self._action_scale = action_scale
        self.height = height
        self.width = width
        self.z_offset = z_offset    # Make sure gripper always higher than table z_offset
        self.xy_tolerance = xy_tolerance
        self.force_orientation = True
        self.tool_target_quat = None

        if 'state' in env_name:
            self._from_pixel = False
        elif 'vision' in env_name:
            self._from_pixel = True
        else:
            raise NotImplementedError('Not support this environment: {}'.format(env_name))

        self.key_obs = 'front_rgb'

        self._reset_tool_pos = np.array([0.25, 0., 1.])

        self._make_env()
        self.env.env._obs_config.front_camera.image_size = (84, 84)

        # Create observation space
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6, ),
            dtype=np.float32
        )

        # Create action space
        self._action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_space.shape[0] - 4 - 1,),
            dtype=np.float32
        )

        # Get configurations of boundary
        self.ws_min, self.ws_max = self.get_range_boundary()

        self.step_cnt = None

    def _make_env(self):
        # Configure Camera
        left_shoulder_camera = CameraConfig(image_size=(self.width, self.height))
        right_shoulder_camera = CameraConfig(image_size=(self.width, self.height))
        wrist_camera = CameraConfig(image_size=(self.width, self.height))
        front_camera = CameraConfig(image_size=(self.width, self.height))
        obs_config = ObservationConfig(left_shoulder_camera=left_shoulder_camera,
                                       right_shoulder_camera=right_shoulder_camera,
                                       wrist_camera=wrist_camera,
                                       front_camera=front_camera)
        self._max_episode_steps = None
        if self._env_name == "reach_target-state-v0":
            self._max_episode_steps = 100

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
        return self.env.env._robot.arm.get_tip().get_position().copy()

    @property
    def _current_tool_quaternion(self):
        # Return quaternion of tip of gripper: (x, y, z, w)
        return self.env.env._robot.arm.get_tip().get_quaternion().copy()

    @property
    def _robot_state_low_dim(self):
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

    def step(self, pos):
        action = self._make_full_action(pos, self._action_scale)
        obs, _, done, info = self.env.step(action)

        # Control done flag by wrapper
        self.step_cnt += 1
        if self.step_cnt == self._max_episode_steps:
            done = True

        success, _ = self.env.task._task.success()
        info.update(dict(success=success))

        rew = self.compute_reward(action, obs)

        return self._get_obs(), rew, done, info

    def compute_reward(self, act, obs, info=None):
        if self.env.task._task.__class__.__name__ == 'ReachTarget':
            success, _ = self.env.task._task.success()
            reward = -np.linalg.norm(self.env.task._task.target.get_position() -
                                     self.env.task._task.robot.arm.get_tip().get_position(), ord=2)
            # final_rew = 0.0 if success else -1.0
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
        _boundary_center = self.env.task._task.boundaries.get_position()
        b_min_x, b_max_x, b_min_y, b_max_y, b_min_z, b_max_z = self.env.task._task.boundaries.get_bounding_box()
        b_min_x, b_max_x = (b_min_x - self.xy_tolerance, b_max_x + self.xy_tolerance)
        b_min_y, b_max_y = (b_min_y - self.xy_tolerance, b_max_y + self.xy_tolerance)
        b_min_z += self.z_offset  # Add this offset to void gripper collide with table
        ws_min = np.round(_boundary_center + np.array([b_min_x, b_min_y, b_min_z]), 2)
        ws_max = np.round(_boundary_center + np.array([b_max_x, b_max_y, b_max_z]), 2)

        return ws_min, ws_max

    def _make_full_action(self, pos, scale=1.0, check_valid=True):
        pos *= scale
        if check_valid:
            pos = self._check_workspace_valid(pos)

        gripper = np.array([1])
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
            obs = self.env.task._scene.get_observation()
            obs = obs.__dict__[self.key_obs]
        else:
            obs_full = self._robot_state_low_dim
            # obs = np.concatenate((obs_full[7:14], obs_full[-3:]))
            obs = np.concatenate((obs_full[22:25], obs_full[-3:]))
        return obs.copy()

    def _check_workspace_valid(self, action):
        # Note: action is delta relative to current position
        # Get expected next position
        tool_next_position = self._current_tool_position + action
        # Make sure next action is valid, i.e.: within workspace (boundaries)
        tool_next_position = np.clip(tool_next_position, self.ws_min, self.ws_max)
        # Get action (delta) after truncateing
        truncated_action = tool_next_position - self._current_tool_position
        if not np.allclose(action, truncated_action, atol=1e-6):
            print('[DEBUG]: Truncating action')
        return truncated_action


if __name__ == '__main__':
    # env = RLBenchWrapper(env_name='reach_target-state-v0', render=True)
    env = RLBenchWrapper_v1(env_name='reach_target-state-v0', render=True)
    obs = env.reset()
    for i in range(100):
        print('step: ', i)
        obs, rew, done, info = env.step(env.action_space.sample())

