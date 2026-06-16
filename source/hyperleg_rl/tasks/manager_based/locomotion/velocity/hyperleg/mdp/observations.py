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


def goal_elapsed_time_s(
    env: "ManagerBasedRLEnv",
    command_name: str,
) -> torch.Tensor:
    """Seconds since the current ``pose_command`` goal was last sampled [s].

    Resets to zero on episode reset and whenever a new goal is resampled after arrival.

    Returns:
        Tensor of shape ``(num_envs, 1)``.
    """
    term = env.command_manager.get_term(command_name)
    return term.goal_elapsed_s.unsqueeze(1)


def _pose_goal_nav_state(
    env: "ManagerBasedRLEnv",
    command_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    term = env.command_manager.get_term(command_name)
    robot: "Articulation" = env.scene[term.cfg.asset_name]
    env_xy = env.scene.env_origins[:, :2]
    base_xy = robot.data.root_pos_w[:, :2] - env_xy
    goal_xy = term.pos_command_w[:, :2] - env_xy
    delta_xy = term.pos_command_w[:, :2] - robot.data.root_pos_w[:, :2]
    dist = torch.norm(delta_xy, dim=1)
    return base_xy, goal_xy, delta_xy, dist, term.cfg.arrival_std


def base_pos_xy(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Robot root XY in env-local frame. Shape ``(num_envs, 2)``."""
    base_xy, _, _, _, _ = _pose_goal_nav_state(env, command_name)
    return base_xy


def goal_pos_xy(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Goal XY in env-local frame. Shape ``(num_envs, 2)``."""
    _, goal_xy, _, _, _ = _pose_goal_nav_state(env, command_name)
    return goal_xy


def goal_delta_xy(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """World-frame XY offset from robot root to goal. Shape ``(num_envs, 2)``."""
    _, _, delta_xy, _, _ = _pose_goal_nav_state(env, command_name)
    return delta_xy


def goal_distance_xy(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """World-frame XY distance to goal [m] (same metric as arrival). Shape ``(num_envs, 1)``."""
    _, _, _, dist, _ = _pose_goal_nav_state(env, command_name)
    return dist.unsqueeze(1)


def goal_distance_normalized(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """``goal_distance_xy / arrival_std``. Shape ``(num_envs, 1)``."""
    _, _, _, dist, arrival_std = _pose_goal_nav_state(env, command_name)
    return (dist / arrival_std).unsqueeze(1)


def inside_arrival_zone(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """``1.0`` when ``goal_distance_xy < arrival_std``, else ``0.0``. Shape ``(num_envs, 1)``."""
    _, _, _, dist, arrival_std = _pose_goal_nav_state(env, command_name)
    return (dist < arrival_std).float().unsqueeze(1)


def cot_components(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Per-leg CoT breakdown as a 4-channel observation [W].

    Columns ``[joule_L, joule_R, mech_L, mech_R]`` — the same components
    that :func:`cost_of_transport_penalty` (rewards.py) uses for ``P`` in
    ``P/v`` (``m·g`` fixed in ``RewTerm.weight``). Exposed to the policy so it can see its own joule and mech
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


def foot_grf_magnitude(
    env: "ManagerBasedRLEnv",
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Per-foot net GRF magnitude in Newtons.

    Vector-sums ``net_forces_w`` over every body in ``left_sensor_cfg`` (e.g.
    ``l_heel`` + ``l_tp``) and ``right_sensor_cfg`` (e.g. ``r_heel`` + ``r_tp``),
    then returns ``[||F_L||, ||F_R||]``.

    Returns:
        Tensor of shape ``(num_envs, 2)``.
    """
    sensor = env.scene.sensors[left_sensor_cfg.name]
    left = sensor.data.net_forces_w[:, left_sensor_cfg.body_ids, :].sum(dim=1)
    right = sensor.data.net_forces_w[:, right_sensor_cfg.body_ids, :].sum(dim=1)
    return torch.stack((
        torch.linalg.norm(left, dim=-1),
        torch.linalg.norm(right, dim=-1),
    ), dim=1)
