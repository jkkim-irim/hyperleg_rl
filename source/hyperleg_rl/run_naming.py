# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HyperLeg training run folder / wandb run naming."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def toe_prefix(task_name: str) -> str:
    """Return ``ToeEquipped`` or ``ToeAblation`` from a registered HyperLeg gym task id."""
    if "-WoToe" in task_name:
        return "ToeAblation"
    if task_name.startswith("HyperLeg-"):
        return "ToeEquipped"
    return "Run"


def _task_param_suffix(
    task_name: str,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> str:
    parts: list[str] = []

    if "Locomotion" in task_name or "Ttest" in task_name:
        terrain_type = env_cfg.scene.terrain.terrain_type
        if terrain_type != "generator":
            parts.append(terrain_type)
        elif env_cfg.scene.terrain.terrain_generator is not None:
            parts.append("rough")

    if agent_cfg.max_iterations != 1000:
        parts.append(f"it{agent_cfg.max_iterations}")

    return "_".join(parts)


def build_hyperleg_run_name(
    task_name: str,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    extra_suffix: str = "",
) -> str:
    """Build ``{ToePrefix}_{YYYY-MM-DD_HH-MM}_{params?}_{extra?}``."""
    segments = [toe_prefix(task_name), datetime.now().strftime("%Y-%m-%d_%H-%M")]

    param_suffix = _task_param_suffix(task_name, env_cfg, agent_cfg)
    if param_suffix:
        segments.append(param_suffix)
    if extra_suffix:
        segments.append(extra_suffix)

    return "_".join(segments)
