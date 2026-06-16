# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HyperLeg-specific command generators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.commands.commands_cfg import UniformPose2dCommandCfg
from isaaclab.envs.mdp.commands.pose_2d_command import UniformPose2dCommand
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ArrivalResamplePose2dCommand(UniformPose2dCommand):
    """Uniform 2D position goals that advance when the robot reaches the current target.

    Timer-based resampling from :class:`CommandTerm` is disabled; a new goal is sampled
    only when ``error_pos_2d < cfg.arrival_std``. The episode continues (no reset).

    Heading is not part of the task: :attr:`command` exposes only ``(pos_x_b, pos_y_b)``.

    Debug visualization draws a sphere of radius :attr:`cfg.arrival_std` at the goal
    (same PointInstancer path as the upstream pose arrow).
    """

    cfg: ArrivalResamplePose2dCommandCfg

    @property
    def command(self) -> torch.Tensor:
        """XY goal offset in base frame. Shape is (num_envs, 2)."""
        return self.pos_command_b[:, :2]

    def __init__(self, cfg: ArrivalResamplePose2dCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._prev_arrived = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._arrived_env_ids = torch.zeros(0, dtype=torch.long, device=self.device)
        self._best_goal_dist = torch.full((self.num_envs,), float("inf"), dtype=torch.float32, device=self.device)
        self.metrics["arrivals"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _reset_best_goal_dist(self, env_ids: Sequence[int] | slice) -> None:
        if isinstance(env_ids, slice):
            self._best_goal_dist[:] = float("inf")
        else:
            self._best_goal_dist[env_ids] = float("inf")

    def _flush_arrivals_for_reset_envs(self, env_ids: Sequence[int] | slice) -> None:
        """Credit arrivals detected in reward compute when the env resets on the same step.

        Env step order is reward → ``_reset_idx`` → ``command_manager.compute()``. Without
        flushing here, ``reset()`` would discard ``_arrived_env_ids`` before ``compute()``
        runs, so ``goal_arrived`` fires but ``metrics/arrivals`` stays 0 and mid-episode
        resample is skipped for terminating envs. Goal resample for those envs is left to
        ``CommandTerm.reset`` (``super().reset``); non-terminating arrivals stay for
        ``compute()``.
        """
        if len(self._arrived_env_ids) == 0:
            return
        if isinstance(env_ids, slice):
            pending = self._arrived_env_ids
        else:
            reset_ids = torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
            pending = self._arrived_env_ids[torch.isin(self._arrived_env_ids, reset_ids)]
        if len(pending) == 0:
            return
        self.metrics["arrivals"][pending] += 1.0
        self._reset_best_goal_dist(pending)
        self._prev_arrived[pending] = False
        keep = ~torch.isin(self._arrived_env_ids, pending)
        self._arrived_env_ids = self._arrived_env_ids[keep]

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        self._flush_arrivals_for_reset_envs(env_ids)
        if isinstance(env_ids, slice):
            self._prev_arrived[:] = False
            self._reset_best_goal_dist(env_ids)
        else:
            self._prev_arrived[env_ids] = False
            self._reset_best_goal_dist(env_ids)
        return super().reset(env_ids)

    def compute(self, dt: float):
        if len(self._arrived_env_ids) > 0:
            arrived_env_ids = self._arrived_env_ids
            self.metrics["arrivals"][arrived_env_ids] += 1.0
            self._resample(arrived_env_ids)
            self._reset_best_goal_dist(arrived_env_ids)
            self._prev_arrived[arrived_env_ids] = False
            self._arrived_env_ids = torch.zeros(0, dtype=torch.long, device=self.device)

        self._update_metrics()
        self._update_command()

    def _resample_command(self, env_ids: Sequence[int]):
        self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
        r = torch.empty(len(env_ids), device=self.device)
        self.pos_command_w[env_ids, 0] += r.uniform_(*self.cfg.ranges.pos_x)
        self.pos_command_w[env_ids, 1] += r.uniform_(*self.cfg.ranges.pos_y)
        self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]

    def _update_command(self):
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)

    def _debug_vis_callback(self, event):
        std = self.cfg.arrival_std
        scales = torch.full((self.num_envs, 3), std, device=self.device)
        self.goal_pose_visualizer.visualize(translations=self.pos_command_w, scales=scales)


@configclass
class ArrivalResamplePose2dCommandCfg(UniformPose2dCommandCfg):
    """Position goal (XY) that resamples on arrival (not on a timer)."""

    class_type: type = ArrivalResamplePose2dCommand

    simple_heading: bool = True

    @configclass
    class Ranges(UniformPose2dCommandCfg.Ranges):
        """XY goal sampling ranges. Heading is unused when ``simple_heading`` is True."""

        heading: tuple[float, float] = (0.0, 0.0)

    arrival_std: float = 0.3
    """World-frame XY distance [m] below which the goal counts as reached."""

    goal_pose_visualizer_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/pose_goal")
    goal_pose_visualizer_cfg.markers["sphere"].radius = 1.0
    goal_pose_visualizer_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.2, 0.95, 0.35)
    )
