# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HyperLeg curriculum terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _learning_iteration(env: ManagerBasedRLEnv, num_steps_per_env: int) -> int:
    return env.common_step_counter // num_steps_per_env


def command_lin_vel_x_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    num_steps_per_env: int,
    iters_per_step: int,
    vel_step: float,
    initial_max: float,
    final_max: float,
    command_name: str = "base_velocity",
) -> float:
    """Expand ``lin_vel_x`` upper bound every ``iters_per_step`` learning iterations."""
    command_term = env.command_manager.get_term(command_name)
    stage = _learning_iteration(env, num_steps_per_env) // iters_per_step
    lin_vel_x_max = min(initial_max + stage * vel_step, final_max)
    lin_vel_x_min, _ = command_term.cfg.ranges.lin_vel_x
    command_term.cfg.ranges.lin_vel_x = (lin_vel_x_min, lin_vel_x_max)
    return lin_vel_x_max


def pose_command_range_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    num_steps_per_env: int,
    iters_per_step: int,
    initial_limit: float,
    final_limit: float,
    limit_step: float,
    warmup_iters: int = 0,
    command_name: str = "pose_command",
) -> float:
    """Expand pose-goal XY sampling range after a warmup, then every ``iters_per_step`` iters.

    Holds ``initial_limit`` for the first ``warmup_iters`` learning iterations, then
    adds ``limit_step`` every ``iters_per_step`` iterations until ``final_limit``.
    """
    command_term = env.command_manager.get_term(command_name)
    learning_iter = _learning_iteration(env, num_steps_per_env)
    if learning_iter < warmup_iters:
        stage = 0
    else:
        stage = (learning_iter - warmup_iters) // iters_per_step
    half_extent = min(initial_limit + stage * limit_step, final_limit)
    pos_range = (-half_extent, half_extent)
    command_term.cfg.ranges.pos_x = pos_range
    command_term.cfg.ranges.pos_y = pos_range
    return half_extent
