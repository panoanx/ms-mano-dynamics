import sys

sys.path.append(".")

import torch.nn as nn
import wandb
import os
import time

from stable_baselines3.ppo.ppo import PPO
from pyrfuniverse.utils.proc_wrapper import SubprocVecEnv
from envs.pd_muscle_env import PMEnv
from data.mim_loader import load_mim_data
from stable_baselines3.common.callbacks import CheckpointCallback
from config.idnet import IDNetConfig

config = IDNetConfig()


def make_env(
    mim_data=load_mim_data(config.mim_data_path),
    **kwargs
):
    return PMEnv(
        mim_data=mim_data,
        num_agents=config.num_agents,
        mt_shape=config.mt_shape,
        graphics=config.graphics,
        executable_file=config.executable_file,
        **kwargs
    )


def create_checkpoint_callback(na):
    net_structure = ""
    for key, value in na.items():
        net_structure += f"{key}-"
        for v in value:
            net_structure += f"{v}-"

    save_path = os.path.join(
        "./models/", "ppo", net_structure[:-1], time.strftime("%Y-%m-%d-%H-%M-%S")
    )

    return CheckpointCallback(
        save_freq=1000,
        save_path=save_path,
    )


if __name__ == "__main__":

    wandb.init()
    if config.use_subproc:
        env_fns = [make_env] * config.n_proc
        env = SubprocVecEnv(env_fns, n_agents=config.num_agents)
    else:
        env = make_env()

    # print(env.observation_space, env.action_space)
    net_arch = dict(pi=config.net_arch.policy, vf=config.net_arch.value_function)
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_epochs=config.n_epochs,
        n_steps=config.n_steps,
        learning_rate=config.learning_rate,
        batch_size=config.n_steps * env.num_envs,
        ent_coef=0.001,
        seed=config.seed,
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            net_arch=[net_arch],
        ),
        tensorboard_log="./tensorboard",
    )

    checkpoint_callback = create_checkpoint_callback(net_arch)

    model.learn(
        total_timesteps=config.total_timesteps,
        # bc_init_epoch=0,
        tb_log_name=config.tb_log_name,
        log_interval=1,
        callback=checkpoint_callback,
    )
