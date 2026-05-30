# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HyperLeg-specific observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .rewards import _compute_cot_components

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def cot_components(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Per-leg CoT breakdown as a 4-channel observation [W].

    Columns ``[joule_L, joule_R, mech_L, mech_R]`` — the same components
    that ``actuator_power_consumption`` (rewards.py) sums into a single
    penalty. Exposed to the policy so it can see its own joule and mech
    contributions directly.

    Returns:
        Tensor of shape ``(num_envs, 4)``.
    """
    joule_L, joule_R, mech_L, mech_R = _compute_cot_components(env, asset_cfg)
    return torch.stack((joule_L, joule_R, mech_L, mech_R), dim=1)


def motor_heat(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Per-motor thermal EMA ``(τ_motor / τ_cont)²``, both legs concatenated.

    Reads the ``motor_heat`` buffer from each ``CoupledLegActuator`` (1st-order
    RC heat proxy; ``1.0`` ≈ steady-state operation at rated continuous
    torque). Canonical role order ``[HY HR HP KN AK FT TO]`` per leg,
    concatenated as ``L || R``.

    Returns:
        Tensor of shape ``(num_envs, 14)`` (or ``(num_envs, 12)`` for the
        toe-ablation variant).
    """
    robot: "Articulation" = env.scene[asset_cfg.name]
    heat_L = robot.actuators["left_leg"].motor_heat
    heat_R = robot.actuators["right_leg"].motor_heat
    return torch.cat((heat_L, heat_R), dim=1)


def feet_air_time(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_tp"),
) -> torch.Tensor:
    """Current swing-phase duration per foot in seconds (0 while in contact).

    Returns ``ContactSensor.data.current_air_time`` for the bodies selected
    by ``sensor_cfg`` — default ``.*_tp`` picks the toe-pad bodies of both
    feet. Increases linearly during swing, resets to 0 at touchdown, so the
    policy gets a direct gait-phase / stride-rhythm signal.

    Returns:
        Tensor of shape ``(num_envs, num_feet)`` (typically ``(E, 2)``).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    air_time = sensor.data.current_air_time
    if sensor_cfg.body_ids is not None and sensor_cfg.body_ids != slice(None):
        air_time = air_time[:, sensor_cfg.body_ids]
    return air_time


def heel_grf_magnitude(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_heel"),
) -> torch.Tensor:
    """Per-heel net contact-force magnitude in Newtons.

    Returns:
        Tensor of shape ``(num_envs, num_heels)`` — typically ``(E, 2)`` with
        ``[||F_l_heel||, ||F_r_heel||]`` in the body-index order resolved by
        ``sensor_cfg``.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w
    if sensor_cfg.body_ids is not None and sensor_cfg.body_ids != slice(None):
        forces = forces[:, sensor_cfg.body_ids, :]
    return torch.linalg.norm(forces, dim=-1)
