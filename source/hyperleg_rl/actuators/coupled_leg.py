# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HyperLeg Cooperative-Actuation actuator with 7×7 Jacobian + joint friction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.actuators.actuator_pd import IdealPDActuator
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    from .coupled_leg_cfg import CoupledLegActuatorCfg


# Canonical joint role order used by the 7×7 Jacobian and motor-side limits.
_ROLE_SUFFIXES: tuple[str, ...] = ("_HY", "_HR", "_HP", "_KN", "_AK", "_FT", "_TO")


class CoupledLegActuator(IdealPDActuator):
    """Per-leg actuator implementing HyperLeg paper eq. (1)–(5).

    Compute pipeline (per substep):

        1. PD on joint position/velocity error → ``joint_τ``
        2. Reorder USD → canonical [HY HR HP KN AK FT TO]
        3. ``motor_vel = J · joint_vel`` ; ``motor_τ = J^{-T} · joint_τ``
        4. Motor-space 4-quadrant clip (τ_stall, ω_0 per motor)
        5. ``joint_τ = J^T · motor_τ_clipped``
        6. Scatter canonical → USD
        7. Subtract joint-space friction: ``τ -= f_c·tanh(q̇/v_m) + f_v·q̇``

    The Jacobian is the standing-pose linearization (paper Table I footnote).
    """

    cfg: "CoupledLegActuatorCfg"

    def __init__(self, cfg: "CoupledLegActuatorCfg", *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)

        if self.num_joints != 7:
            raise ValueError(
                "CoupledLegActuator requires exactly 7 joints per leg "
                f"(got {self.num_joints} for joints {self._joint_names})."
            )

        # Map canonical role index -> USD index, by joint-name suffix.
        role_to_usd: list[int] = []
        for suffix in _ROLE_SUFFIXES:
            matches = [i for i, name in enumerate(self._joint_names) if name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one joint ending with {suffix!r}, "
                    f"got {matches} from {self._joint_names}."
                )
            role_to_usd.append(matches[0])
        self._role_order = torch.tensor(role_to_usd, dtype=torch.long, device=self._device)

        # Jacobian J = ∂m/∂q, canonical order. Cache forward/inverse and transposes.
        if self.cfg.jacobian_joint_to_motor is None:
            raise ValueError("jacobian_joint_to_motor must be provided.")
        J = torch.tensor(self.cfg.jacobian_joint_to_motor, dtype=torch.float32, device=self._device)
        if J.shape != (7, 7):
            raise ValueError(f"jacobian_joint_to_motor must be 7×7, got {tuple(J.shape)}.")
        self._J = J
        self._J_T = J.T.contiguous()
        self._J_inv = torch.linalg.inv(J)

        # Sanity: J · J^{-1} ≈ I_7
        eye = torch.eye(7, dtype=J.dtype, device=self._device)
        if not torch.allclose(self._J @ self._J_inv, eye, atol=1e-4):
            raise ValueError("Provided jacobian_joint_to_motor is numerically singular.")

        # Motor-side limits (Nm, rad/s). Parsed in USD order, then re-indexed to canonical.
        # τ_cont = continuous torque cap, τ_stall = stall (peak rhombus) torque, ω_0 = no-load speed.
        motor_effort_usd = self._parse_joint_parameter(self.cfg.motor_effort_limit, 0.0)
        motor_sat_usd = self._parse_joint_parameter(self.cfg.motor_saturation_effort, 0.0)
        motor_velocity_usd = self._parse_joint_parameter(self.cfg.motor_velocity_limit, 0.0)
        self._motor_effort_limit = motor_effort_usd[:, self._role_order].contiguous()
        self._motor_saturation_effort = motor_sat_usd[:, self._role_order].contiguous()
        self._motor_velocity_limit = motor_velocity_usd[:, self._role_order].contiguous()

        # Corner velocity ω_c = ω_0 · (1 + τ_cont/τ_stall) where the rhombus would otherwise descend
        # below -τ_cont. Beyond ω_c, joint motion is exogenous (back-drive); we clamp ω to ±ω_c
        # before evaluating the rhombus, matching DCMotor._clip_effort.
        self._motor_vel_at_effort_lim = self._motor_velocity_limit * (
            1.0 + self._motor_effort_limit / self._motor_saturation_effort
        )

        # Joint-side friction params (USD order — applied directly to joint_vel).
        self._friction_coulomb = self._parse_joint_parameter(self.cfg.friction_coulomb, 0.0)
        self._activation_vel = self._parse_joint_parameter(self.cfg.activation_vel, float("inf"))
        self._friction_viscous = self._parse_joint_parameter(self.cfg.friction_viscous, 0.0)

        # Diagnostic buffers (canonical [HY HR HP KN AK FT TO] order). Updated every compute().
        # Useful for verifying CA: τ_j_KN ≈ -31.11·sum(last_motor_tau[3:]) when friction is small.
        self.last_motor_tau = torch.zeros(self._num_envs, 7, device=self._device)
        self.last_motor_vel = torch.zeros_like(self.last_motor_tau)

    def reset(self, env_ids: Sequence[int]) -> None:
        pass

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        # 1. Joint-space PD
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        joint_tau = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts
        self.computed_effort = joint_tau

        # 2. Reorder USD → canonical [HY HR HP KN AK FT TO]
        idx = self._role_order
        joint_tau_c = joint_tau.index_select(1, idx)
        joint_vel_c = joint_vel.index_select(1, idx)

        # 3. joint → motor (J = ∂m/∂q):
        #      motor_vel = J · joint_vel   (batched rows: joint_vel @ J^T)
        #      motor_τ   = J^{-T} · joint_τ (batched rows: joint_τ @ J_inv)
        motor_vel = joint_vel_c @ self._J_T
        motor_tau = joint_tau_c @ self._J_inv

        # 4. Motor-space 4-quadrant clip
        motor_tau_clipped = self._clip_motor_4quadrant(motor_tau, motor_vel)
        self.last_motor_tau[:] = motor_tau_clipped
        self.last_motor_vel[:] = motor_vel

        # 5. motor → joint: τ_j = J^T · τ_m  (batched rows: τ_m @ J)
        joint_tau_c_out = motor_tau_clipped @ self._J

        # 6. Scatter canonical → USD
        joint_tau_out = torch.zeros_like(joint_tau)
        joint_tau_out.index_copy_(1, idx, joint_tau_c_out)

        # 7. Joint-space friction (paper eq. 5)
        friction = self._friction_coulomb * torch.tanh(joint_vel / self._activation_vel) + (
            self._friction_viscous * joint_vel
        )
        joint_tau_out = joint_tau_out - friction

        # Hand off as effort-only command.
        self.applied_effort = joint_tau_out
        control_action.joint_efforts = joint_tau_out
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def _clip_motor_4quadrant(self, motor_tau: torch.Tensor, motor_vel: torch.Tensor) -> torch.Tensor:
        """Linear DC motor torque-speed hexagonal clip (paper eq. 4).

        Mirrors ``isaaclab.actuators.DCMotor._clip_effort`` exactly, but applied
        in motor space using motor-side stall / continuous / no-load values.
        Shape collapses to a rhombus when ``saturation_effort == effort_limit``.
        """
        tau_stall = self._motor_saturation_effort
        tau_cont = self._motor_effort_limit
        omega_0 = self._motor_velocity_limit
        # Clamp motor velocity into the rhombus's valid range so torque limits stay sensible
        # when external back-drive pushes ω beyond the corner velocity.
        omega = torch.clip(motor_vel, min=-self._motor_vel_at_effort_lim, max=self._motor_vel_at_effort_lim)
        torque_speed_top = tau_stall * (1.0 - omega / omega_0)
        torque_speed_bottom = tau_stall * (-1.0 - omega / omega_0)
        max_effort = torch.clip(torque_speed_top, max=tau_cont)
        min_effort = torch.clip(torque_speed_bottom, min=-tau_cont)
        return torch.clip(motor_tau, min=min_effort, max=max_effort)
