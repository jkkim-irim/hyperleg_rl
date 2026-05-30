# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HyperLeg-specific MDP terms (rewards, observations, terminations)."""

from .observations import (
    cot_components,
    feet_air_time,
    heel_grf_magnitude,
    motor_heat,
)
from .rewards import (
    actuator_power_consumption,
    heel_grf_l1,
    motor_thermal_penalty,
)

__all__ = [
    "actuator_power_consumption",
    "cot_components",
    "feet_air_time",
    "heel_grf_l1",
    "heel_grf_magnitude",
    "motor_heat",
    "motor_thermal_penalty",
]
