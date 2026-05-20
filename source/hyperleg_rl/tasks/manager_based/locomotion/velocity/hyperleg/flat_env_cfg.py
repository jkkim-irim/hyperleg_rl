# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HyperLeg biped locomotion environment configuration (flat terrain)."""

from isaaclab.utils.configclass import configclass

from .rough_env_cfg import HyperLegRoughEnvCfg


@configclass
class HyperLegFlatEnvCfg(HyperLegRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # rewards
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class HyperLegFlatEnvCfg_PLAY(HyperLegFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
