import gym
from gym import core, spaces
import rlbench.gym
import numpy as np


ENV_SUPPORT = ['reach_target-state-v0', 'reach_target-vision-v0']


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
                 action_scale=0.1,
                 height=128,
                 width=128,
                 min_z_offset=0.05):
        assert env_name in ENV_SUPPORT, 'Environment {} is not supported'.format(env_name)

        self._env_name = env_name
        self._render = render
        self._action_scale = action_scale
        self._max_episode_steps = 100
        self.height = height
        self.width = width
        self.min_z_offset = min_z_offset

        if 'state' in env_name:
            self._from_pixel = False
        elif 'vision' in env_name:
            self._from_pixel = True
        else:
            raise NotImplementedError('Not support this environment: {}'.format(env_name))

        self.key_obs = 'front_rgb'

        # self._reset_tool_pos = np.array([0.125, 0., 1.])
        self._reset_tool_pos = np.array([0.25, 0., 1.])
        # Create task
        self._make_env()

        # Create observation space
        # self._observation_space = self.env.observation_space
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

        _boundary_center = self.env.task._task.boundaries.get_position()
        b_min_x, b_max_x, b_min_y, b_max_y, b_min_z, b_max_z = self.env.task._task.boundaries.get_bounding_box()
        b_min_z += min_z_offset     # Add this offset to void gripper collide with table
        self.ws_min = np.round(_boundary_center + np.array([b_min_x, b_min_y, b_min_z]), 2)
        self.ws_max = np.round(_boundary_center + np.array([b_max_x, b_max_y, b_max_z]), 2)

    def _make_env(self):
        render_mode = 'human' if self._render else None
        self.env = gym.make(self._env_name, render_mode=render_mode)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def _current_tool_position(self):
        return self.env.env._robot.arm.get_tip().get_position().copy()

    @property
    def _robot_state_low_dim(self):
        return self.env.task._scene.get_observation().get_low_dim_data().copy()

    def reset(self):
        self.env.reset()
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

    def _make_full_action(self, pos, scale=1.0, check_valid=True):
        pos *= scale
        if check_valid:
            pos = self._check_workspace_valid(pos)

        gripper = np.array([1])
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

