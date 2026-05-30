# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Cfg for HyperLeg's Cooperative-Actuation 7×7 Jacobian actuator."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.actuators.actuator_pd_cfg import IdealPDActuatorCfg
from isaaclab.utils import configclass

from . import coupled_leg


@configclass
class CoupledLegActuatorCfg(IdealPDActuatorCfg):
    """Per-leg actuator: motor↔joint via 7×7 Jacobian + 4-quadrant clip.

    Canonical order: [HY, HR, HP, KN, AK, FT, TO]. ``joint_names_expr``
    must contain exactly these 7 joints (any order — actuator reorders
    by suffix). PD gains and ``effort/velocity_limit`` stay joint-side;
    the 4-quadrant clip uses ``motor_*`` limits.
    """

    class_type: type = coupled_leg.CoupledLegActuator

    jacobian_joint_to_motor: tuple[tuple[float, ...], ...] = MISSING
    """``J = ∂motor/∂joint``, 7×7. See ``allex_isaac_twin/docs/gain_transform_ref``."""

    motor_effort_limit: dict[str, float] | float = MISSING
    """Motor continuous torque [Nm] — horizontal cap of 4-quadrant clip."""

    motor_saturation_effort: dict[str, float] | float = MISSING
    """Motor stall torque [Nm]. ``> effort_limit`` → hexagonal clip; ``=`` → rhombus."""

    motor_velocity_limit: dict[str, float] | float = MISSING
    """Motor no-load speed [rad/s]."""

    friction_coulomb: dict[str, float] | float | None = None
    """Joint-side coulomb friction [Nm]."""

    activation_vel: dict[str, float] | float = float("inf")
    """tanh smoothing scale for coulomb friction [rad/s]. ``+inf`` disables it."""

    friction_viscous: dict[str, float] | float | None = None
    """Joint-side viscous friction [Nm·s/rad]."""

    thermal_time_constant: float = 3.0
    """EMA time constant [s] for ``motor_heat = ⟨(τ_m/τ_cont)²⟩``. Symmetric heating/cooling."""
