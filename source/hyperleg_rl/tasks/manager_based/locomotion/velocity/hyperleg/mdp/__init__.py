# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HyperLeg-specific MDP terms (rewards, observations, terminations)."""

from .commands import ArrivalResamplePose2dCommandCfg
from .curriculums import (
    command_lin_vel_x_levels,
    pose_command_range_levels,
)
from .observations import (
    base_pos_xy,
    cot_components,
    feet_air_time,
    foot_grf_magnitude,
    goal_delta_xy,
    goal_distance_normalized,
    goal_distance_xy,
    goal_pos_xy,
    heel_grf_magnitude,
    inside_arrival_zone,
    motor_heat,
)
from .rewards import (
    backward_base_lin_vel_x_hinge,
    cost_of_transport_penalty,
    goal_arrived_event,
    lateral_base_lin_vel_y_hinge,
    motor_thermal_penalty,
    pose_command_progress_reward,
)
from .terminations import non_unicycle_base_vel_violation

__all__ = [
    "ArrivalResamplePose2dCommandCfg",
    "backward_base_lin_vel_x_hinge",
    "base_pos_xy",
    "cost_of_transport_penalty",
    "command_lin_vel_x_levels",
    "pose_command_range_levels",
    "cot_components",
    "feet_air_time",
    "foot_grf_magnitude",
    "goal_arrived_event",
    "goal_delta_xy",
    "goal_distance_normalized",
    "goal_distance_xy",
    "goal_pos_xy",
    "heel_grf_magnitude",
    "inside_arrival_zone",
    "lateral_base_lin_vel_y_hinge",
    "motor_heat",
    "motor_thermal_penalty",
    "non_unicycle_base_vel_violation",
    "pose_command_progress_reward",
]
