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


# Canonical motor-role order — mirrors CoupledLegActuator._ROLE_SUFFIXES.
_CANONICAL_ROLES: tuple[str, ...] = ("HY", "HR", "HP", "KN", "AK", "FT", "TO")


def motor_thermal_overuse(
    env: "ManagerBasedRLEnv",
    threshold: float | dict[str, float] = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Hinge penalty above per-motor thermal limit, in fold-of-threshold units.

    Reads ``motor_heat`` (EMA of ``(τ_motor / τ_cont)²``) from both leg
    actuators. ``motor_heat == 1.0`` means sustained operation at the rated
    continuous torque (steady-state thermal limit).

    Penalty per motor:

        max(0, motor_heat / threshold − 1)²

    "Fold over threshold" rather than raw overshoot, so motors with very
    different absolute heat scales (e.g. KN ~0.22 vs TO ~0.006 at steady
    state) contribute equally to the gradient when each is the same fold
    above its own limit. Per-joint thresholds let each motor be policed
    around its real operating envelope instead of a single number that
    only KN/HP ever approach.

    ``threshold`` may be:

    * ``float`` — same threshold for every role (in the new normalized form).
    * ``dict[str, float]`` — keyed by canonical role
      ``{"HY","HR","HP","KN","AK","FT","TO"}``. Roles must cover the
      actuator's joint count (6 for the toe-ablation variant, no ``TO``).

    Returns:
        Per-env scalar in ``[0, ∞)``. Sign comes from ``RewTerm.weight``.
    """
    robot: "Articulation" = env.scene[asset_cfg.name]
    heat_L = robot.actuators["left_leg"].motor_heat
    heat_R = robot.actuators["right_leg"].motor_heat

    if isinstance(threshold, dict):
        roles = _CANONICAL_ROLES[: heat_L.shape[1]]
        thr = heat_L.new_tensor([threshold[r] for r in roles])
    else:
        thr = float(threshold)

    over_L = (heat_L / thr - 1.0).clamp(min=0.0).square()
    over_R = (heat_R / thr - 1.0).clamp(min=0.0).square()
    return over_L.sum(dim=1) + over_R.sum(dim=1)


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
