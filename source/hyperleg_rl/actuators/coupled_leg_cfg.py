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
    """One actuator per HyperLeg leg (7 joints).

    Implements paper eq. (1)–(5):
    motor↔joint torque/velocity transform via a constant 7×7 Jacobian
    ``J = ∂m/∂q`` followed by motor-space 4-quadrant clip, with joint-space
    coulomb·tanh + viscous friction.

    Canonical joint order for both J and ``motor_*`` limits:
        [HY, HR, HP, KN, AK, FT, TO]

    ``joint_names_expr`` must match exactly seven joints whose names end with
    ``_HY / _HR / _HP / _KN / _AK / _FT / _TO`` (one each), e.g.
    ``["L_HY", "L_HR", "L_HP", "L_KN", "L_AK", "L_FT", "L_TO"]``.

    PD gains (``stiffness`` / ``damping``) stay in joint space — no gain
    transform is applied. ``effort_limit`` / ``velocity_limit`` are joint-side
    PhysX safety bounds; the 4-quadrant clip uses ``motor_*`` instead.
    """

    class_type: type = coupled_leg.CoupledLegActuator

    jacobian_joint_to_motor: tuple[tuple[float, ...], ...] = MISSING
    """``J = ∂motor/∂joint``, 7×7, dimensionless. Canonical order
    [HY, HR, HP, KN, AK, FT, TO]. Convention matches the reference doc
    ``allex_isaac_twin/docs/gain_transform_ref/Stiffness_Transform.md``."""

    motor_effort_limit: dict[str, float] | float = MISSING
    """Motor-side **continuous** torque [Nm], per joint (the horizontal cap
    in paper Fig. 4(c) / IsaacLab DCMotor's ``effort_limit``)."""

    motor_saturation_effort: dict[str, float] | float = MISSING
    """Motor-side **stall (peak) torque** [Nm], per joint — torque at zero
    speed on the linear torque-speed line (paper τ_stall / IsaacLab DCMotor's
    ``saturation_effort``). With current amplification, ``saturation_effort
    > effort_limit`` produces the hexagonal clip shape; if equal, the clip
    collapses to a rhombus."""

    motor_velocity_limit: dict[str, float] | float = MISSING
    """Motor-side no-load speed [rad/s], per joint. ω_0 in paper eq. (4)."""

    friction_coulomb: dict[str, float] | float | None = None
    """Joint-side coulomb friction magnitude [Nm], per joint."""

    activation_vel: dict[str, float] | float = float("inf")
    """Joint velocity scale [rad/s] for the smooth-sign tanh in the friction
    model. Effectively disables coulomb friction when ``+inf``."""

    friction_viscous: dict[str, float] | float | None = None
    """Joint-side viscous friction [Nm·s/rad], per joint."""
