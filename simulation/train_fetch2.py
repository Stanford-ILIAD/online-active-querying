from util.rlkit_custom import rollout
import os

from rlkit.torch.pytorch_util import set_gpu_mode

import torch
import rlkit
from rlkit.core import logger
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS
import tqdm

import numpy as np
import make_env
import pickle5 as pickle

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

pretrain_data = pickle.load(open('base-50.pkl', 'rb'))




def experiment(variant):
    env = make_env.make_push()
    expl_env = NormalizedBoxEnv(env)
    eval_env = NormalizedBoxEnv(env)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    run_name = 'push-' + ''.join(random.choice(string.ascii_lowercase) for _ in range(12))
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs={
            "batch_size": 128,
            "max_path_length": 500,
            "max_path_length": 500,
            "min_num_steps_before_training": 3300,
            "num_epochs": 2000,
            "num_eval_steps_per_epoch": 2500,
            "num_expl_steps_per_train_loop": 2500,
            "num_trains_per_train_loop": 1000
        },
        trainer_kwargs={
            "discount": 0.99,
            "policy_lr": 0.001,
            "qf_lr": 0.0005,
            "reward_scale": 1.0,
            "soft_target_tau": 0.005,
            "target_update_period": 5,
            "use_automatic_entropy_tuning": True
        },
    )
    setup_logger(run_name, variant=variant, log_dir=f'logs/{run_name}')
    experiment(variant)

