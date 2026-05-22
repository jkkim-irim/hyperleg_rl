# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HyperLeg-specific MDP terms (rewards, observations, terminations)."""

from .rewards import motor_thermal_overuse, motor_torque_symmetry

__all__ = ["motor_thermal_overuse", "motor_torque_symmetry"]
