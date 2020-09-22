from gym import core, spaces
import gym
import numpy as np

class RLbenchWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        obs_filters='front_rgb',
        action_type='position',
        action_space=3,
        frame_skip=1,
        heigh=84,
        width=84,
        channels_first=True,
        action_scale=0.05
    ):
        super(RLbenchWrapper, self).__init__(env)
        self.obs_filters=obs_filters
        self.action_type=action_type
        # self.rlbench_env=env
        self._action_scale = action_scale
        self._reset_tool_pos = np.array([0.25, 0., 1.])
        self.channels_first= channels_first
        low_dim_data=['joint_velocities', 'joint_positions',
                     'joint_forces',
                     'gripper_pose', 'gripper_joint_positions',
                     'gripper_touch_forces', 'task_low_dim_state']

        high_dim_data=['left_shoulder_rgb', 'left_shoulder_depth', 'left_shoulder_mask',
                       'right_shoulder_rgb', 'right_shoulder_depth', 'right_shoulder_mask',
                       'wrist_rgb', 'wrist_depth', 'wrist_mask', 'front_rgb', 'front_depth', 'front_mask']
        
        self.observation_space={}
        for elem in self.obs_filters:
            if elem in low_dim_data or elem in high_dim_data:
                # breakpoint()
                if self.env._observation_mode=='vision':
                    full_obs=self.get_obs()
                    # breakpoint()
                    if channels_first==True and not 'depth' in elem and not 'mask' in elem:
                        t=full_obs.__dict__[elem].shape
                        shp=np.array([t[2], t[0], t[1]])
                    else:
                        shp=full_obs.__dict__[elem].shape
                    self.observation_space[elem]= spaces.Box(
                        low=0, high=255, shape=shp, dtype=np.int32)
                elif self.env._observation_mode=='state':
                    if 'shp' not in locals():
                        shp=[]
                    full_obs=self.get_obs()
                    shp+=list(full_obs.__dict__[elem].shape)
                
            else: 
                raise ValueError("Not correct observation space data provided")

        if self.env._observation_mode=='pixel' and len(self.observation_space)==1:
                self.observation_space=self.observation_space.popitem()[1]
        elif self.env._observation_mode=='pixel' and len (self.observation_space)>1:
            self.observation_space=spaces.Dict(self.observation_space)
        if self.env._observation_mode=='state':
            self.observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=[sum(shp)], dtype=np.float32
            )
        
              

        
        #TODO action_space as input to wrapper
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_space.shape[0] - 4 - 1,),
            dtype=np.float32
        )
        _boundary_center = self.env.task._task.boundaries.get_position()
        b_min_x, b_max_x, b_min_y, b_max_y, b_min_z, b_max_z = self.env.task._task.boundaries.get_bounding_box()
        self.ws_min = np.round(_boundary_center + np.array([b_min_x, b_min_y, b_min_z]), 2)
        self.ws_max = np.round(_boundary_center + np.array([b_max_x, b_max_y, b_max_z]), 2)

    def _current_tool_position(self):
        return self.env.env._robot.arm.get_tip().get_position().copy()

    def get_observations(self):
        if self.env._observation_mode=="state":
            full_obs=self.get_obs()
            observation=[]
            for elem in self.obs_filters:
                observation.append(full_obs.__dict__[elem])
            return np.concatenate(observation)
        elif self.env._observation_mode=="vision":
            full_obs=self.get_obs()
            observation={}
            for elem in self.obs_filters:
                visual_data=full_obs.__dict__[elem]
                if self.channels_first and not 'depth' in elem and not 'mask' in elem:
                    img=np.transpose(visual_data,(2,0,1))
                else:
                    img=visual_data.copy()
                observation[elem]=img
            if len(observation)==1:
                observation=observation.popitem()[1]
            return observation

    def _make_full_action(self, pos, scale=1.0, check_valid=True):
        pos *= scale
        if check_valid:
            pos = self._check_workspace_valid(pos)
        # breakpoint()
        #TODO full position + orentation    
        if self.action_type=='position':
            gripper = np.array([1])
            quat = np.array([0, 0, 0, 1])
            action_full = np.concatenate((pos, quat, gripper))
        else:
            action_full = pos
        return action_full

    def _check_workspace_valid(self, action):
        # Note: action is delta relative to current position
        # Get expected next position
        tool_next_position = self._current_tool_position() + action
        # Make sure next action is valid, i.e.: within workspace (boundaries)
        tool_next_position = np.clip(tool_next_position, self.ws_min, self.ws_max)
        # Get action (delta) after truncateing
        truncated_action = tool_next_position - self._current_tool_position()
        if not np.allclose(action, truncated_action, atol=1e-6):
            print('[DEBUG]: Truncating action')
        return truncated_action
        
    def reset(self):
        self.env.reset()
        # This loop for move gripper to initial position
        while True:
            del_pos = self._reset_tool_pos - self._current_tool_position()
            act = self._make_full_action(del_pos, 0.05, False)
            _, _, done, _ = self.env.step(act)
            if done:
                self.env.reset()
                continue
            if np.linalg.norm(del_pos) < 1e-3 and not done:
                break

        return self.get_observations()

    def compute_reward(self, act, obs, info=None):
        success, _ = self.env.task._task.success()
        reward = self.env.task._task.reward()

        # final_rew = reward + float(success)
        # final_rew = 0.0 if success else -1.0
        final_rew = reward
        return final_rew


    def step(self, pos):
        action = self._make_full_action(pos, self._action_scale)
        _, _, done, info = self.env.step(action)
        observation=self.get_observations()
        rew = self.compute_reward(action, observation)
        return observation, rew, done, info

