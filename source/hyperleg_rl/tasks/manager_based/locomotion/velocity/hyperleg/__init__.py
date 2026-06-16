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

##
# End-to-end goal-reaching locomotion ("goto"): one policy walks + navigates from scratch.
##

gym.register(
    id="HyperLeg-Goto-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goto_env_cfg:HyperLegGotoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HyperLegGotoPPORunnerCfg",
    },
)

gym.register(
    id="HyperLeg-Goto-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goto_env_cfg:HyperLegGotoEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HyperLegGotoPPORunnerCfg",
    },
)

gym.register(
    id="HyperLeg-Goto-WoToe-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goto_env_cfg:HyperLegGotoWoToeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HyperLegGotoPPORunnerCfg",
    },
)

gym.register(
    id="HyperLeg-Goto-WoToe-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goto_env_cfg:HyperLegGotoWoToeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HyperLegGotoPPORunnerCfg",
    },
)

##
# End-to-end point-goal navigation ("T test"): one policy walks + navigates from scratch.
#
# Training log / wandb run names (see hyperleg_rl.run_naming):
#   ToeEquipped_*  — HyperLeg-Ttest-v0, HyperLeg-Ttest-Rough-v0, HyperLeg-Ttest-Play-v0, …
#   ToeAblation_*  — HyperLeg-Ttest-WoToe-v0, HyperLeg-Ttest-WoToe-Rough-v0, …
##

gym.register(
    id="HyperLeg-Ttest-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Ttest_env_cfg:HyperLegTtestEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HyperLegTtestPPORunnerCfg",
    },
)

gym.register(
    id="HyperLeg-Ttest-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Ttest_env_cfg:HyperLegTtestEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HyperLegTtestPPORunnerCfg",
    },
)

gym.register(
    id="HyperLeg-Ttest-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Ttest_env_cfg:HyperLegTtestRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HyperLegTtestPPORunnerCfg",
    },
)

gym.register(
    id="HyperLeg-Ttest-WoToe-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Ttest_env_cfg:HyperLegTtestWoToeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HyperLegTtestPPORunnerCfg",
    },
)

gym.register(
    id="HyperLeg-Ttest-WoToe-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Ttest_env_cfg:HyperLegTtestWoToeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HyperLegTtestPPORunnerCfg",
    },
)

gym.register(
    id="HyperLeg-Ttest-WoToe-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Ttest_env_cfg:HyperLegTtestWoToeRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HyperLegTtestPPORunnerCfg",
    },
)
