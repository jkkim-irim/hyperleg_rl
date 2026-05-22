# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HyperLeg-specific reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def motor_thermal_overuse(
    env: "ManagerBasedRLEnv",
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Hinge penalty above per-motor thermal limit.

    Reads ``motor_heat`` (EMA of ``(τ_motor / τ_cont)²``) from both leg
    actuators. ``motor_heat == 1.0`` means sustained operation at the rated
    continuous torque (steady-state thermal limit). Penalty is quadratic
    above ``threshold``, zero below — a "cliff" that lets the policy briefly
    exceed rating but punishes sustained overload.

    Returns:
        Per-env scalar in ``[0, ∞)``. Sign comes from ``RewTerm.weight``.
    """
    robot: "Articulation" = env.scene[asset_cfg.name]
    heat_L = robot.actuators["left_leg"].motor_heat
    heat_R = robot.actuators["right_leg"].motor_heat
    excess_L = (heat_L - threshold).clamp(min=0.0).square()
    excess_R = (heat_R - threshold).clamp(min=0.0).square()
    return excess_L.sum(dim=1) + excess_R.sum(dim=1)


def motor_torque_symmetry(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quadratic penalty on bilateral load asymmetry per motor role.

    Reads ``motor_load_avg`` (EMA of ``|τ_motor| / τ_cont``, ~4-s window) from
    both leg actuators and squares the L−R difference for each canonical
    motor role ``[HY HR HP KN AK FT TO]``. ``|τ|`` is used (not signed τ)
    because walking has mirrored sign profiles between legs; signed means
    would cancel for a normal gait.

    Returns:
        Per-env scalar in ``[0, ∞)``. Sign comes from ``RewTerm.weight``.
    """
    robot: "Articulation" = env.scene[asset_cfg.name]
    load_L = robot.actuators["left_leg"].motor_load_avg
    load_R = robot.actuators["right_leg"].motor_load_avg
    return (load_L - load_R).square().sum(dim=1)
