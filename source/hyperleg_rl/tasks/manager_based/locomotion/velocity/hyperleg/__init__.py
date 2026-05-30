# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HyperLeg biped velocity-tracking locomotion task."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="HyperLeg-Locomotion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hyperleg_env_cfg:HyperLegEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HyperLegPPORunnerCfg",
    },
)

gym.register(
    id="HyperLeg-Locomotion-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hyperleg_env_cfg:HyperLegEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HyperLegPPORunnerCfg",
    },
)

gym.register(
    id="HyperLeg-Locomotion-WoToe-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hyperleg_env_cfg:HyperLegWoToeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HyperLegPPORunnerCfg",
    },
)

gym.register(
    id="HyperLeg-Locomotion-WoToe-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hyperleg_env_cfg:HyperLegWoToeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HyperLegPPORunnerCfg",
    },
)
