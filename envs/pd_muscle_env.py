from collections import OrderedDict
from copy import deepcopy
from gym import spaces
import numpy as np

from stable_baselines3.common.vec_env.util import obs_space_info
from .unity_env import UnityEnv


class PMEnv(UnityEnv):
    """
    PMEnv class represents the environment for the PD Muscle control algorithm.
    The PMEnv manages multiple processes and transforms the process-level observations and rewards into global variables (n_proc * num_agents).


    Args:
        mt_shape (tuple): The shape of the muscle torque array. Default is (90, 3).
        **kwargs: Additional keyword arguments to be passed to the UnityEnv superclass.

    Attributes:
        observation_space (gym.Space): The observation space of the environment.
        action_space (gym.Space): The action space of the environment.
        keys (list): The keys of the observation space.
        shapes (dict): The shapes of the observation space.
        dtypes (dict): The data types of the observation space.
        buf_obs (collections.OrderedDict): The buffer for storing observations.
        buf_dones (numpy.ndarray): The buffer for storing done flags.
        buf_rews (numpy.ndarray): The buffer for storing rewards.
        buf_infos (list): The buffer for storing additional information.
        num_envs (int): The number of environments.
        muscle_count (int): The number of muscles.
        worker_idxs (list): The indices of the workers.

    Methods:
        step(actions): Performs a step in the environment.
        get_obs(agent_idx): Returns the observations for a specific agent.
        get_reward(obs, agent_idx): Calculates the reward for a specific agent.
    """

    def __init__(
        self,
        mt_shape=(90, 3),
        **kwargs,
    ):
        super().__init__(
            use_muscle=False,
            use_pd=True,
            mt_shape=mt_shape,
            aug=True,
            **kwargs,
        )
        self.observation_space = spaces.Dict(
            {
                "target_joint_bias": spaces.Box(
                    -np.inf, np.inf, shape=(15, 3), dtype=np.float32
                ),
                "target_joint_vel": spaces.Box(
                    -np.inf, np.inf, shape=(15, 3), dtype=np.float32
                ),
                "curr_joint_bias": spaces.Box(
                    -np.inf, np.inf, shape=(15, 3), dtype=np.float32
                ),
                "curr_joint_vel": spaces.Box(
                    -np.inf, np.inf, shape=(15, 3), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(31,), dtype=np.float32
        )  # muscle activations

        self.keys, self.shapes, dtypes = obs_space_info(self.observation_space)
        self.buf_obs = OrderedDict(
            [
                (k, np.zeros((self.num_envs,) + tuple(self.shapes[k]), dtype=dtypes[k]))
                for k in self.keys
            ]
        )
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.set_muscles(np.zeros((self.num_envs, self.muscle_count)), self.worker_idxs)
        self._step()

    def step(self, actions):
        """
        Performs a step in the environment.

        Args:
            actions (numpy.ndarray): The actions to take in the environment.

        Returns:
            tuple: A tuple containing the updated observations, rewards, done flags, and additional information.
        """
        actions = actions.reshape((self.num_envs, self.muscle_count))
        actions = actions / 2 + 0.5  # [0, 1]

        for agent_idx in self.worker_idxs:  # idxs of workers start from 1
            frame_id = (self.t[agent_idx] + self.mim_offset[agent_idx]) % self.mim_len
            mim_frame = self.mim_data["jb"][frame_id]  # jb: joint bias, the dof values
            if self.AUG:
                prev_mim_frame = self.mim_data["jb"][frame_id - 1]
                mim_frame += np.random.normal(0, 0.1, size=mim_frame.shape) * (
                    mim_frame - prev_mim_frame
                )
            self.set_pd_target(
                mim_frame,  # shape (15, 3)
                [agent_idx],
            )
            self.mim_frame = mim_frame
        self.pd_control(
            idxs=self.worker_idxs,
        )  # unity forward
        self.set_muscles(actions, self.worker_idxs)

        self._step()

        for agent_idx in self.worker_idxs:
            (
                obs,
                self.buf_rews[agent_idx],
                self.buf_dones[agent_idx],
                self.buf_infos[agent_idx],
            ) = self.agent_step(agent_idx=agent_idx)
            if self.buf_dones[agent_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[agent_idx]["terminal_observation"] = obs
                obs = self.agent_reset(agent_idx=agent_idx)
            self._save_obs(agent_idx, obs)

        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def get_obs(self, agent_idx):
        """
        Returns the observations for a specific agent.

        Args:
            agent_idx (int): The index of the agent.

        Returns:
            dict: A dictionary containing the observations for the agent.
        """
        muscle_segment_torque = self.muscle_segment_torque[agent_idx]
        muscle_joint_torque = self.muscle_joint_torque[
            agent_idx
        ]  # muscle's torque on a specific joint
        pd_torque = self.pd_torque[agent_idx]
        joint_bias = self.joint_bias[agent_idx]
        joint_vel = self.joint_vel[agent_idx]
        prev_joint_bias = self.prev_joint_bias[agent_idx]
        prev_joint_vel = self.prev_joint_vel[agent_idx]

        obs_dict = {
            "muscle_segment_torque": muscle_segment_torque,
            "muscle_joint_torque": muscle_joint_torque,
            "pd_torque": pd_torque,
            "target_joint_bias": joint_bias,
            "target_joint_vel": joint_vel,
            "curr_joint_bias": prev_joint_bias,
            "curr_joint_vel": prev_joint_vel,
        }

        return obs_dict

    def get_reward(self, obs, agent_idx):
        """
        Calculates the reward for a specific agent.

        Args:
            obs (dict): The observations for the agent.
            agent_idx (int): The index of the agent.

        Returns:
            float: The reward for the agent.
        """
        diff = -np.linalg.norm(
            self.muscle_joint_torque[agent_idx] - self.prev_pd_torque[agent_idx]
        )
        reward = np.exp(diff * 0.1)
        if np.isinf(reward):
            reward = 0
        return reward
