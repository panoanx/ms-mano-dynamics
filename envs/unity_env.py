from collections import OrderedDict
from copy import deepcopy
import numpy as np
from gym import spaces
from gym.utils import seeding
from pyrfuniverse.envs.base_env import RFUniverseBaseEnv
from pyrfuniverse.side_channel import (
    IncomingMessage,
    OutgoingMessage,
)

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.util import obs_space_info

MUSCLE_MASK = 0b00000001
PD_MASK = 0b00000010


class UnityEnv(RFUniverseBaseEnv, DummyVecEnv):
    """
    UnityEnv is a class that represents the Unity environment for reinforcement learning.
    Each UnityEnv <-> a Unity process. It handles multiple in-process agents (num_agents).

    Args:
        executable_file (str): The path to the Unity executable file.
        num_agents (int): The number of agents in the environment.
        mt_shape (tuple): The shape of the muscle torque array.
        joint_indexs (range): The range of joint indices.
        mim_data (None or dict): The MIM data.
        proc_id (int): The process ID.
        graphics (bool): Whether to enable graphics.
        aug (bool): Whether to enable data augmentation.
        use_muscle (bool): Whether to use muscle.
        use_pd (bool): Whether to use PD control.

    Attributes:
        mim_len (int): The length of the MIM data.
        mim_offset (list): The offset of the MIM data.
        mt_shape (tuple): The shape of the muscle torque array.
        muscle_count (int): The number of muscles.
        joint_indexs (range): The range of joint indices.
        AUG (bool): Whether data augmentation is enabled.
        num_envs (int): The number of environments.
        t (list): The timestep for each agent.
        muscle_segment_torque (ndarray): The muscle segment torque array.
        muscle_joint_torque (ndarray): The muscle joint torque array.
        pd_torque (ndarray): The PD torque array.
        joint_bias (ndarray): The joint bias array.
        joint_vel (ndarray): The joint velocity array.
        prev_pd_torque (ndarray): The previous PD torque array.
        prev_joint_bias (ndarray): The previous joint bias array.
        prev_joint_vel (ndarray): The previous joint velocity array.
        target_joint_ang_vel (ndarray): The target joint angular velocity array.
        worker_idxs (list): The indices of the worker agents.
        pd_constant (float): The PD constant.
        muscle_scaler (float): The muscle scaler.
        muscle_constant (float): The muscle constant.
    """

    def __init__(
        self,
        executable_file=None,
        num_agents=1,
        mt_shape=(90, 3),
        joint_indexs=range(1, 16),
        mim_data=None,
        proc_id=0,
        graphics=False,
        aug=False,
        use_muscle=False,
        use_pd=True,
    ):
        super().__init__(
            proc_id=proc_id,
            log_level=0,
            graphics=graphics,
            executable_file=executable_file,
        )
        self.SetTimeScale(1)
        self.mim_data = mim_data
        if mim_data is not None:
            self.mim_len = mim_data["len"]
            self.mim_offset = (
                np.random.randint(0, self.mim_len, size=num_agents) * aug
            ).tolist()  # if data augmentation is on, offset is random
        else:
            self.mim_len = 1
            self.mim_offset = np.zeros(num_agents).tolist()
        self.mt_shape = mt_shape
        self.muscle_count = 31
        self.joint_indexs = joint_indexs
        self.AUG = aug

        self.num_envs = num_agents
        self.t = np.zeros(num_agents).astype(int).tolist()
        muscle_mask = MUSCLE_MASK * use_muscle  # should use ~ here
        pd_mask = PD_MASK * use_pd  # but python's binary literal sucks
        mask = muscle_mask | pd_mask
        self.SendObject("SetMode", mask)
        self.SendObject("SetNumAgent", num_agents)
        self._step()
        self._step()
        self.AddListenerObject(
            "CollectMuscleSegmentTorque", self._collect_muscle_segment_torque
        )
        self.AddListenerObject(
            "CollectMuscleJointTorque", self._collect_muscle_joint_torque
        )
        self.AddListenerObject("CollectPDTorque", self._collect_pd_torque)
        self.AddListenerObject("CollectJointBias", self._collect_joint_bias)
        self.AddListenerObject("CollectJointAngVel", self._collect_joint_ang_vel)
        self.AddListenerObject(
            "CollectTargetJointAngVel", self._collect_target_joint_ang_vel
        )
        self.pd_torque = np.zeros((self.num_envs,) + (15, 3))
        self.muscle_segment_torque = np.zeros((self.num_envs,) + mt_shape)
        self.muscle_joint_torque = np.zeros((self.num_envs,) + (15, 3))
        self.joint_bias = np.zeros((self.num_envs,) + (15, 3))
        self.mim_frame = self.joint_bias
        self.joint_vel = np.zeros((self.num_envs,) + (15, 3))
        self.prev_pd_torque = np.zeros_like(self.pd_torque)
        self.prev_joint_bias = self.joint_bias
        self.prev_joint_vel = self.joint_vel
        self.target_joint_ang_vel = self.joint_vel

        self.worker_idxs = np.arange(self.num_envs).tolist()

        self.pd_constant = 461.08872269202044
        self.muscle_scaler = 90.05
        self.muscle_constant = 1.0

    def step(self, actions):
        raise NotImplementedError

    def get_reward(self, obs, agent_idx):
        raise NotImplementedError

    def reset(self):
        for agent_idx in self.worker_idxs:
            obs = self.agent_reset(agent_idx=agent_idx)
            self._save_obs(agent_idx, obs)
        return self._obs_from_buf()

    def unity_step(self, actions, ret, i):
        """Sends a step signal to the simulator. The simulator will instantly step forward by one frame.

        Args:
            actions (np.ndarray): The actions.
            ret (np.ndarray): The return array.
            i (int): The index.
        """
        ret[i] = self.step(actions)

    def unity_reset(self, ret, i):
        """
        Resets the Unity environment and stores the result in the given list.

        Parameters:
        ret (list): The list to store the reset result.
        i (int): The index at which to store the reset result.

        Returns:
        None
        """
        ret[i] = self.reset()

    def agent_step(self, agent_idx=0):
        """
        Executes a single step for the agent (gym level step / step related to RL algorithm).

        Args:
            agent_idx (int): The index of the agent.

        Returns:
            tuple: A tuple containing the following elements:
                - obs (object): The observation received from the environment.
                - reward (float): The reward received from the environment.
                - done (bool): A flag indicating whether the episode is done.
                - info (dict): Additional information about the step.
        """
        self.t[agent_idx] += 1

        obs = self.get_obs(agent_idx)
        done = False
        info = {"is_success": False}
        reward = self.get_reward(obs, agent_idx=agent_idx)

        done = self.t[agent_idx] >= self.mim_len

        return obs, reward, done, info

    def agent_reset(self, agent_idx):
        """
        Resets the agent's state and returns the initial observation.

        Parameters:
            agent_idx (int): The index of the agent to reset.

        Returns:
            observation (object): The initial observation of the agent.

        """
        # self.env.reset()

        self.set_pose(self.mim_data["rot"][0], agent_idxs=[agent_idx])
        self.agent_step(agent_idx=agent_idx)
        self.t[agent_idx] = 0

        return self.get_obs(agent_idx=agent_idx)

    def seed(self, seed=1234):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_attr_data(self, agent_idx, key):
        data_list = list()
        for joint_index in self.joint_indexs:
            try:
                data = np.array(self.GetAttr(joint_index + agent_idx * 1000).data[key])
                if key == "position":
                    data -= np.array([agent_idx, 0, 0])
                data_list.append(data)
            except BaseException as e:
                pass
        return np.concatenate(data_list)

    def get_obs(self, agent_idx):
        muscle_segment_torque = self.muscle_segment_torque[agent_idx]
        muscle_joint_torque = self.muscle_joint_torque[agent_idx]
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

        # self.p.append(pd_torque)
        # self.m.append(self.muscle_joint_torque[env_idx])
        # print(np.array(self.p).std(), np.array(self.m).std())

        # obs_dict['timestep'] = (self.t[env_idx] - self.mim_frame_range.start) / \
        # (self.mim_frame_range.stop - self.mim_frame_range.start)

        return obs_dict

    def pd_control(self, steps=1, interpolate=0, idxs=None):
        if idxs is None:
            idxs = self.worker_idxs

        for _ in range(steps):
            self.SendObject("ApplyPDTorqueAll", idxs, interpolate)
            # self._step()  # unity step

    def set_pose(self, rot, agent_idxs=[0], ifReg=False, representation="euler"):
        """set pose of the agent

        Args:
            rot (np.ndarray): rotation matrix
            agent_idxs (list): list of agent index
            ifReg (bool): if use regularization
            representation (str): 'euler' or 'axis_angle'
        """
        # only euler is supported for regularization
        assert ifReg == False or representation == "euler"

        # self.env.reset()
        # for env_idx in env_idxs:
        #     for joint_index in self.joint_indexs:
        #         self.GetAttr(joint_index + env_idx * 1000).SetTransform(
        #             rotation=list(rot[joint_index - 1]),
        #             is_world=False,
        #         )
        for agent_idx in agent_idxs:
            self.SendObject(
                "SetPose", agent_idx, rot.reshape(-1).tolist(), ifReg, representation
            )

    def set_ang_vel(self, ang_vel, agent_idxs=[0]):
        for agent_idx in agent_idxs:
            self.SendObject("SetAngVel", agent_idx, ang_vel.reshape(-1).tolist())

    def fake_simulate(self):
        self.SendObject("FakeSimulate")

    def set_pose_by_pd(self, rot, agent_idxs=[0], ifReg=False, representation="euler"):
        assert ifReg == False or representation == "euler"
        for agent_idx in agent_idxs:
            self.SendObject(
                "SetPoseByPD",
                agent_idx,
                rot.reshape(-1).tolist(),
                ifReg,
                representation,
            )

    def set_pd_target(self, rot, agent_idxs=[0], ifReg=True):
        for agent_idx in agent_idxs:
            self.SendObject("SetPDTarget", agent_idx, rot.reshape(-1).tolist(), ifReg)

    def set_muscles(self, a: np.ndarray, idxs):
        if a.ndim == 1:
            a = a.reshape((1,) + a.shape)  # [0, 1]
        for i, idx in enumerate(idxs):
            self.SendObject(
                "SetMuscles",
                idx,
                a[i].tolist(),
            )
        return

    # the _collect functions are called when the listener receives the corresponding message. This is for handling data from unity.

    def _collect_muscle_segment_torque(self, obj: list):
        num_envs = obj[0]

        # muscle envs, excluding 0 (skeleton only)
        assert num_envs == len(self.worker_idxs)

        muscle_segment_torque = np.array(obj[1]).reshape((num_envs,) + self.mt_shape)
        self.muscle_segment_torque = muscle_segment_torque * self.muscle_constant

    def _collect_muscle_joint_torque(self, obj: list):
        num_envs = obj[0]

        # muscle envs, excluding 0 (skeleton only)
        assert num_envs == len(self.worker_idxs)

        muscle_joint_torque = np.array(obj[1]).reshape((num_envs,) + (15, 3))
        self.muscle_joint_torque = muscle_joint_torque * self.muscle_scaler

    def _collect_pd_torque(self, obj: list):
        num_envs = obj[0]

        # muscle envs, excluding 0 (skeleton only)
        assert num_envs == len(self.worker_idxs)

        pd_torque = np.array(obj[1]).reshape((num_envs,) + (15, 3))
        self.prev_pd_torque = self.pd_torque
        self.pd_torque = pd_torque * self.pd_constant

    def _collect_joint_bias(self, obj: list):
        num_envs = obj[0]
        assert num_envs == len(self.worker_idxs)  # env 0 is mimicker, not worker
        joint_bias = np.array(obj[1]).reshape((num_envs,) + (15, 3))
        self.prev_joint_bias = self.joint_bias
        self.joint_bias = joint_bias

    def _collect_joint_ang_vel(self, obj: list):
        num_envs = obj[0]
        assert num_envs == len(self.worker_idxs)
        joint_vel = np.array(obj[1]).reshape((num_envs,) + (15, 3))
        self.prev_joint_vel = self.joint_vel
        self.joint_vel = joint_vel

    def _collect_target_joint_ang_vel(self, obj: list):
        num_envs = obj[0]
        assert num_envs == len(self.worker_idxs)
        target_joint_ang_vel = np.array(obj[1]).reshape((num_envs,) + (15, 3))
        self.target_joint_ang_vel = target_joint_ang_vel
