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


# Per-motor electrical power coefficient Cp = R_LL / K_t² [W/Nm²] keyed by
# canonical role (6-step trapezoidal convention, motor-side torque).
# Verified from datasheets:
#   RI60: R_LL=0.9 Ω, K_t=0.085 Nm/A → 0.9/0.085² = 124.567474 ✓
# Paper Table I has U10 and RI60 values swapped (typo); the legacy
# POWER_COEF values were correct — datasheet wins.
_POWER_COEF_BY_ROLE: dict[str, float] = {
    "HY": 3.54148162,
    "HR": 3.54148162,
    "HP": 3.54148162,
    "KN": 33.44203328,
    "AK": 124.567474,
    "FT": 124.567474,
    "TO": 124.567474,
}


def _get_power_coef(env: "ManagerBasedRLEnv", n: int, device: torch.device) -> torch.Tensor:
    """Lazy-cached POWER_COEF tensor in canonical role order ``[:n]``."""
    cached = getattr(env, "_cot_power_coef", None)
    if cached is not None and cached.shape[0] == n and cached.device == device:
        return cached
    coef = torch.tensor(
        [_POWER_COEF_BY_ROLE[r] for r in _CANONICAL_ROLES[:n]],
        dtype=torch.float32,
        device=device,
    )
    env._cot_power_coef = coef
    return coef


def _compute_cot_components(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy CoT breakdown ``(joule_L, joule_R, mech_L, mech_R)`` per env [W].

    Reproduces ``actuator_torques_l2`` (legacy ``rewards.py``) using the
    motor-space tensors that ``CoupledLegActuator`` already caches in
    canonical ``[HY HR HP KN AK FT TO]`` order — so no Jacobian re-application
    is needed here. For each leg:

    * ``joule = Σ POWER_COEF · τ_m²`` — winding I²R surrogate from RAW
      motor torque (I ∝ τ_m so I²R ∝ τ_m²; driver-side electrical losses
      are charged separately on the mech term where they physically belong).
    * ``P_raw = τ_m · ω_m`` then
      ``P_mech = where(P_raw < 0, P_raw·0.8, P_raw/0.9)`` — η=0.8 regen on
      back-drive, η=0.9 motoring on positive power (paper-faithful). The
      sign of P_raw correctly identifies motoring (Q1/Q3) vs regen (Q2/Q4)
      regardless of the sign of τ_m alone.
    * ``mech = Σ P_mech``.

    Both 7-DoF (full) and 6-DoF (toe-ablation) actuators are supported.
    """
    robot: "Articulation" = env.scene[asset_cfg.name]
    left = robot.actuators["left_leg"]
    right = robot.actuators["right_leg"]

    tau_L = left.last_motor_tau
    tau_R = right.last_motor_tau
    vel_L = left.last_motor_vel
    vel_R = right.last_motor_vel

    pc = _get_power_coef(env, tau_L.shape[1], tau_L.device)

    # Step B+: raw motor torque for Joule and mech (no tau asymmetric scaling).
    joule_L = (pc * tau_L.square()).sum(dim=1)
    joule_R = (pc * tau_R.square()).sum(dim=1)

    p_L = tau_L * vel_L
    p_R = tau_R * vel_R
    # eta_regen=0.8 (paper), eta_out=0.9 (paper).
    p_L_s = torch.where(p_L < 0, p_L * 0.8, p_L / 0.9)
    p_R_s = torch.where(p_R < 0, p_R * 0.8, p_R / 0.9)

    mech_L = p_L_s.sum(dim=1)
    mech_R = p_R_s.sum(dim=1)

    return joule_L, joule_R, mech_L, mech_R


def _battery_input_power(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Battery-input power [W] = ``joule_L + joule_R + mech_L + mech_R``."""
    joule_L, joule_R, mech_L, mech_R = _compute_cot_components(env, asset_cfg)
    return joule_L + joule_R + mech_L + mech_R


def cost_of_transport_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    velocity_min: float = 0.1,
) -> torch.Tensor:
    """Instantaneous ``P / v`` [W·s/m] — proportional to transport CoT.

    ``P`` is battery-input power (joule + signed mech). ``v`` is horizontal
    transport speed [m/s]: body-frame ``|v · cmd_hat|`` when
    ``||cmd_xy|| > 0.05`` m/s, else ``||v_xy||_b``. Fixed ``m·g`` (HyperLeg
    ≈32 kg) is absorbed into ``RewTerm.weight``; divide by ``m·g`` at eval for
    dimensionless CoT ``⟨P⟩/(m·g·⟨v⟩)``. Sign comes from ``RewTerm.weight``.
    """
    robot: "Articulation" = env.scene[asset_cfg.name]
    power = _battery_input_power(env, asset_cfg)

    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    vel_xy = robot.data.root_lin_vel_b[:, :2]
    cmd_speed = torch.linalg.norm(cmd_xy, dim=1)
    cmd_hat = cmd_xy / cmd_speed.unsqueeze(1)
    v_proj = (vel_xy * cmd_hat).sum(dim=1).abs()
    v_xy = torch.linalg.norm(vel_xy, dim=1)
    v = torch.where(cmd_speed > 0.05, v_proj, v_xy)

    return power / v.clamp(min=velocity_min)


def motor_thermal_penalty(
    env: "ManagerBasedRLEnv",
    threshold: float | dict[str, float] = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """CBF-based thermal penalty: react before exceeding the threshold.

    Defines safety margin ``h = h_th - motor_heat`` (positive = safe) and
    penalizes violations of the CBF condition ``ḣ + γ·h ≥ 0``, i.e.

        penalty = max(0, d/dt[motor_heat] - γ·(h_th - motor_heat))

    The CBF gain is derived from the actuator's thermal time constant
    ``γ = 1/τ_EMA``, so no additional hyperparameter is introduced beyond the
    per-motor ``threshold`` already used by :func:`motor_thermal_overuse`.

    Behavior:

    * Heat far below threshold, slow change → 0 (safe).
    * Heat near threshold → even slow heating triggers penalty.
    * Heat above threshold → only cooling fast enough escapes penalty.

    State (``env._thermal_cbf_prev``) is created lazily and persists across
    steps; on the first call (or after a reset that zeros ``motor_heat``),
    the negative-rate transient is clipped by ``clamp(min=0)`` and produces
    no false penalty.

    ``threshold`` accepts the same forms as :func:`motor_thermal_overuse`:

    * ``float`` — same for every motor.
    * ``dict[str, float]`` — keyed by canonical role ``{"HY","HR","HP","KN","AK","FT","TO"}``.

    Returns:
        Per-env scalar in ``[0, ∞)``. Sign comes from ``RewTerm.weight``.
    """
    robot: "Articulation" = env.scene[asset_cfg.name]
    dt = env.step_dt
    prev_store = env.__dict__.setdefault("_thermal_cbf_prev", {})

    total_penalty = torch.zeros(env.num_envs, device=env.device)
    for side in ("left_leg", "right_leg"):
        actuator = robot.actuators[side]
        heat = actuator.motor_heat
        heat_prev = prev_store.get(side)
        if heat_prev is None or heat_prev.shape != heat.shape:
            heat_prev = heat  # first call: zero derivative, zero penalty

        if isinstance(threshold, dict):
            roles = _CANONICAL_ROLES[: heat.shape[1]]
            h_th = heat.new_tensor([threshold[r] for r in roles])
        else:
            h_th = float(threshold)

        gamma = 1.0 / actuator.cfg.thermal_time_constant  # γ = 1/τ_EMA
        heat_dot = (heat - heat_prev) / dt
        violation = heat_dot - gamma * (h_th - heat)
        total_penalty = total_penalty + violation.clamp(min=0.0).sum(dim=1)

        prev_store[side] = heat.detach().clone()

    return total_penalty


def lateral_base_lin_vel_y_hinge(
    env: "ManagerBasedRLEnv",
    margin: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Hinge L2 penalty on base lateral speed above ``margin`` [m/s]."""
    robot: "Articulation" = env.scene[asset_cfg.name]
    excess = robot.data.root_lin_vel_b[:, 1].abs() - margin
    return torch.square(excess.clamp(min=0.0))


def backward_base_lin_vel_x_hinge(
    env: "ManagerBasedRLEnv",
    margin: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Hinge L2 penalty on base backward speed above ``margin`` [m/s]."""
    robot: "Articulation" = env.scene[asset_cfg.name]
    excess = -robot.data.root_lin_vel_b[:, 0] - margin
    return torch.square(excess.clamp(min=0.0))


def pose_command_progress_reward(
    env: "ManagerBasedRLEnv",
    command_name: str,
) -> torch.Tensor:
    """SimTool-style dense reward: ``max(d* - d, 0)`` when XY goal distance improves.

    ``d*`` is tracked per env on ``ArrivalResamplePose2dCommand._best_goal_dist`` and
    resets on episode reset and goal resample. One-shot arrival bonus stays in
    :func:`goal_arrived_event`.
    """
    term = env.command_manager.get_term(command_name)
    robot: "Articulation" = env.scene[term.cfg.asset_name]
    d = torch.norm(term.pos_command_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    d_star = term._best_goal_dist
    first = torch.isinf(d_star)
    dense = torch.where(first, torch.zeros_like(d), torch.clamp(d_star - d, min=0.0))
    term._best_goal_dist = torch.where(first, d, torch.minimum(d_star, d))
    return dense


def goal_arrived_event(
    env: "ManagerBasedRLEnv",
    command_name: str,
) -> torch.Tensor:
    """One-shot 1.0 on the step the robot enters the goal ``arrival_std`` (0.0 otherwise).

    Reward runs before ``command_manager.compute()`` in the env step. Edge-triggered env ids
    are stored on the command term; ``ArrivalResamplePose2dCommand.compute()`` resamples for
    continuing episodes, and ``ArrivalResamplePose2dCommand.reset()`` flushes the same ids when
    the episode ends on the arrival step (before ``compute()``).
    """
    term = env.command_manager.get_term(command_name)
    robot: "Articulation" = env.scene[term.cfg.asset_name]
    dist = torch.norm(term.pos_command_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    arrived = dist < term.cfg.arrival_std
    newly_arrived = arrived & ~term._prev_arrived
    event = newly_arrived.float()
    term._prev_arrived = arrived.clone()
    term._arrived_env_ids = newly_arrived.nonzero(as_tuple=False).flatten()
    return event

