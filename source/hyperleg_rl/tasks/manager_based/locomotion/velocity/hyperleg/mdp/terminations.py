# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HyperLeg-specific termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def non_unicycle_base_vel_violation(
    env: "ManagerBasedRLEnv",
    backward_vel_limit: float = 0.1,
    lateral_vel_limit: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when base backward or lateral speed exceeds its limit [m/s]."""
    robot: "Articulation" = env.scene[asset_cfg.name]
    lin_vel_b = robot.data.root_lin_vel_b
    lateral = lin_vel_b[:, 1].abs() > lateral_vel_limit
    backward = lin_vel_b[:, 0] < -backward_vel_limit
    return lateral | backward


def pose_command_goal_reached(
    env: "ManagerBasedRLEnv",
    command_name: str,
) -> torch.Tensor:
    """End the episode when world-frame XY distance to the pose goal is below ``arrival_std``."""
    term = env.command_manager.get_term(command_name)
    robot: "Articulation" = env.scene[term.cfg.asset_name]
    dist = torch.norm(term.pos_command_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    return dist < term.cfg.arrival_std
