# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class HyperLegPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 200
    experiment_name = "hyperleg_locomotion"
    logger = "wandb"
    wandb_project = "hyperleg_locomotion"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HyperLegGotoPPORunnerCfg(HyperLegPPORunnerCfg):
    """End-to-end goal-reaching locomotion (same big actor as the walker)."""

    experiment_name = "hyperleg_goto"
    wandb_project = "hyperleg_goto"


@configclass
class HyperLegTtestPPORunnerCfg(HyperLegPPORunnerCfg):
    """End-to-end T-test navigation (same actor size as locomotion / goto)."""

    max_iterations = 3500
    save_interval = 200
    experiment_name = "hyperleg_ttest"
    wandb_project = "hyperleg_ttest"
