# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained HyperLeg T-test navigation policy through a fixed T-shaped path.

The policy was trained on ``HyperLeg-Ttest-v0`` with random ``pose_command`` goals
(:class:`ArrivalResamplePose2dCommand`). At play time this script overrides that
command with a **fixed sequence** of five world-frame waypoints (relative to each
env origin), tracing a T:

    start (0, 0)  -- stem -->  (2.5, 0)
                  -- crossbar -->  (2.5, -1.5) -> (2.5, +1.5) -> (2.5, 0)
                  -- return -->  (0, 0)

Waypoints (in order):
    1. (9.14,  0.0)   stem tip
    2. (9.14,  4.57)   crossbar right end
    3. (9.14, -4.57)   crossbar left end
    4. (9.14,  0.0)   crossbar center
    5. (0.0,  0.0)   home

Each control step the live ``pose_command`` term is set to the current waypoint
before ``env.get_observations()`` so the policy sees the correct goal. Arrival
at a waypoint triggers the patched ``_resample_command`` to advance to the next
fixed goal (no random resampling). On home arrival the trial completes and
segment / velocity / path-deviation metrics are logged.

Optional ``--log_csv`` writes per-trial power CSVs plus accumulated
``motor_tw.csv`` / ``joint_tw.csv`` under ``logs/Ttest/<variant>_<ts>/``.

Example:
    ./isaaclab.sh -p projects/hyperleg_rl/scripts/rsl_rl/play_T_test.py \\
        --task HyperLeg-Ttest-Play-v0 --load_run 2026-06-11_02-25-40 \\
        --load_run 2026-06-11_02-25-40 \\
        --checkpoint model_14999.pt --num_envs 1 --trials 3
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a HyperLeg T-test policy on a fixed T-shaped waypoint path.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="HyperLeg-Ttest-Play-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--telemetry",
    action="store_true",
    default=False,
    help="Open a live plot of joint torque / motor heat / motor load_avg for env 0.",
)
parser.add_argument("--arrival_std", type=float, default=None, help="Goal arrival radius [m] (default: env cfg 0.2).")
parser.add_argument("--dev_max", type=float, default=0.5, help="Path deviation [m] that maps the trace color to full red.")
parser.add_argument("--disable_debug_draw", action="store_true", default=False, help="Turn off the debug_draw reference path and trajectory trace.")
parser.add_argument("--trials", type=int, default=10, help="Number of completed-T trials to log, then exit.")
parser.add_argument(
    "--log_csv",
    action="store_true",
    default=True,
    help="Log per-trial power + ankle tau-omega CSVs to logs/Ttest/<variant>_<ts>/.",
)
parser.add_argument("--no-log_csv", action="store_false", dest="log_csv", help="Disable CSV logging.")
parser.add_argument("--log_t_start", type=float, default=0.0, help="Episode-time [s] when power logging opens.")
parser.add_argument("--log_t_end", type=float, default=30.0, help="Episode-time [s] when power logging closes.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
import hyperleg_rl.tasks  # noqa: F401 — register hyperleg gym envs
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Project-local logs root: <hyperleg_rl repo>/logs
_PROJECT_LOGS_ROOT = Path(__file__).resolve().parents[2] / "logs"

# "T"-shaped path, waypoints relative to each env origin (world frame), in order.
_T_WAYPOINTS = (
    (9.14, 0.0),
    (9.14, 4.57),
    (9.14, -4.57),
    (9.14, 0.0),
    (0.0, 0.0),
)

# Reference-path debug line, world frame, drawn through node (0,0) + _T_WAYPOINTS.
_REF_LINE_Z = 1.0
_REF_LINE_COLOR = (0.6, 1.0, 0.2, 1.0)  # RGBA
_REF_LINE_WIDTH = 5.0  # pixels

# Trajectory trace: torso world position every control step; color = T-polyline deviation.
_TRACE_SIZE = 6.0  # pixels
_DEV_CMAP = (
    (0.00, (0.40, 1.00, 0.80)),  # mint (on the line)
    (0.25, (0.10, 0.90, 0.10)),  # green
    (0.50, (1.00, 1.00, 0.00)),  # yellow
    (0.75, (1.00, 0.55, 0.00)),  # orange
    (1.00, (1.00, 0.00, 0.00)),  # red (>= dev_max off the line)
)

_METRICS_COLS = (
    "trial",
    "total_time_s",
    "seg0_s",
    "seg1_s",
    "seg2_s",
    "seg3_s",
    "seg4_s",
    "max_vel_x",
    "mean_vel_x",
    "mean_path_dev_m",
    "max_path_dev_m",
)


def _dev_to_rgba(t: float) -> tuple:
    """Map a normalized deviation ``t`` in [0, 1] to an RGBA color along ``_DEV_CMAP``."""
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    for (t0, c0), (t1, c1) in zip(_DEV_CMAP, _DEV_CMAP[1:]):
        if t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return (c0[0] + f * (c1[0] - c0[0]), c0[1] + f * (c1[1] - c0[1]), c0[2] + f * (c1[2] - c0[2]), 1.0)
    return (*_DEV_CMAP[-1][1], 1.0)


def _dist_to_polyline_xy(px: torch.Tensor, py: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    """Minimum distance from each (px, py) to the polyline through ``nodes`` (S, 2)."""
    min_d = torch.full_like(px, float("inf"))
    for i in range(nodes.shape[0] - 1):
        ax, ay = nodes[i, 0], nodes[i, 1]
        bx, by = nodes[i + 1, 0], nodes[i + 1, 1]
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab2 = abx * abx + aby * aby
        t = torch.where(ab2 > 1e-12, ((apx * abx + apy * aby) / ab2).clamp(0.0, 1.0), torch.zeros_like(px))
        cx = ax + t * abx
        cy = ay + t * aby
        d = torch.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        min_d = torch.minimum(min_d, d)
    return min_d


class SequentialPoseCommandController:
    """Drive ``pose_command`` through a fixed waypoint list instead of random goals.

    Patches the live :class:`ArrivalResamplePose2dCommand` instance so
    ``_resample_command`` advances to the next fixed waypoint on arrival.
    """

    def __init__(self, env, waypoints: tuple[tuple[float, float], ...], arrival_std: float | None = None):
        self.robot = env.unwrapped.scene["robot"]
        self.cmd_term = env.unwrapped.command_manager.get_term("pose_command")
        self.device = env.unwrapped.device
        num_envs = env.unwrapped.num_envs

        if arrival_std is not None:
            self.cmd_term.cfg.arrival_std = float(arrival_std)

        origin_xy = env.unwrapped.scene.env_origins[:, :2]  # (N, 2)
        wp = torch.tensor(waypoints, dtype=torch.float32, device=self.device)  # (W, 2)
        self.goals_w = origin_xy.unsqueeze(1) + wp.unsqueeze(0)  # (N, W, 2)
        self.num_wp = wp.shape[0]

        self.wp_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.finished = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.path_dev = torch.zeros(num_envs, device=self.device)

        nodes_rel = torch.tensor([(0.0, 0.0), *waypoints], dtype=torch.float32, device=self.device)
        self.nodes_w = origin_xy.unsqueeze(1) + nodes_rel.unsqueeze(0)  # (N, S+1, 2)

        self.cmd_term._resample_command = self._sequential_resample_command
        self._set_waypoint(torch.arange(num_envs, device=self.device), 0)

    def _env_ids_tensor(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.wp_idx.shape[0], device=self.device)
        return torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)

    def _set_waypoint(self, env_ids: torch.Tensor, wp_index: int) -> None:
        """Write world-frame goal ``wp_index`` and simple-heading into the command term."""
        self.cmd_term.pos_command_w[env_ids, 0] = self.goals_w[env_ids, wp_index, 0]
        self.cmd_term.pos_command_w[env_ids, 1] = self.goals_w[env_ids, wp_index, 1]
        self.cmd_term.pos_command_w[env_ids, 2] = self.robot.data.default_root_state[env_ids, 2]

        target_vec = self.cmd_term.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
        target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        flipped_target_direction = math_utils.wrap_to_pi(target_direction + torch.pi)
        curr_to_target = math_utils.wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
        curr_to_flipped = math_utils.wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()
        self.cmd_term.heading_command_w[env_ids] = torch.where(
            curr_to_target < curr_to_flipped,
            target_direction,
            flipped_target_direction,
        )

    def _sequential_resample_command(self, env_ids: Sequence[int] | slice) -> None:
        """Advance to the next fixed waypoint when the robot arrives at the current goal."""
        env_ids_t = self._env_ids_tensor(env_ids)
        active = env_ids_t[self.wp_idx[env_ids_t] < self.num_wp]
        if active.numel() == 0:
            return
        self.wp_idx[active] += 1
        self.finished |= self.wp_idx >= self.num_wp
        for eid in active.tolist():
            idx = self.wp_idx[eid].item()
            if idx < self.num_wp:
                self._set_waypoint(torch.tensor([eid], device=self.device), idx)

    def on_reset(self, done_ids: torch.Tensor) -> None:
        """Restart the fixed path for envs that were just reset."""
        if done_ids.numel() > 0:
            self.wp_idx[done_ids] = 0
            self.finished[done_ids] = False
            self.cmd_term._prev_arrived[done_ids] = False
            self.cmd_term._arrived_env_ids = torch.zeros(0, dtype=torch.long, device=self.device)
            self._set_waypoint(done_ids, 0)

    def inject(self) -> None:
        """Refresh base-frame pose command from the current robot state."""
        self.cmd_term._update_command()

    def update_path_dev(self) -> None:
        """Per-env distance from root XY to the T polyline (world frame)."""
        pos = self.robot.data.root_pos_w[:, :2]
        for i in range(pos.shape[0]):
            self.path_dev[i] = _dist_to_polyline_xy(
                pos[i, 0].unsqueeze(0), pos[i, 1].unsqueeze(0), self.nodes_w[i]
            )[0]


class TTtestTrialRecorder:
    """Collect per-trial navigation metrics and append rows to ``ttest_trials.csv``."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self._vel_x_samples: list[float] = []
        self._dev_samples: list[float] = []
        self._segment_arrival_times: list[float] = []
        self._trial_start_t = 0.0
        self._last_wp_idx = 0
        self._header_written = False

    def on_trial_start(self, t_ep_s: float = 0.0) -> None:
        self._vel_x_samples.clear()
        self._dev_samples.clear()
        self._segment_arrival_times.clear()
        self._trial_start_t = t_ep_s
        self._last_wp_idx = 0

    def on_step(self, t_ep_s: float, vel_x: float, path_dev: float, wp_idx: int) -> None:
        self._vel_x_samples.append(vel_x)
        self._dev_samples.append(path_dev)
        if wp_idx > self._last_wp_idx:
            for _ in range(self._last_wp_idx, wp_idx):
                self._segment_arrival_times.append(t_ep_s)
            self._last_wp_idx = wp_idx

    def finalize(self, trial_id: int, t_ep_s: float) -> dict[str, float | int]:
        seg_times: list[float] = []
        prev_t = self._trial_start_t
        for arr_t in self._segment_arrival_times:
            seg_times.append(arr_t - prev_t)
            prev_t = arr_t
        while len(seg_times) < 5:
            seg_times.append(float("nan"))

        return {
            "trial": trial_id,
            "total_time_s": t_ep_s,
            "seg0_s": seg_times[0],
            "seg1_s": seg_times[1],
            "seg2_s": seg_times[2],
            "seg3_s": seg_times[3],
            "seg4_s": seg_times[4],
            "max_vel_x": max(self._vel_x_samples),
            "mean_vel_x": sum(self._vel_x_samples) / len(self._vel_x_samples),
            "mean_path_dev_m": sum(self._dev_samples) / len(self._dev_samples),
            "max_path_dev_m": max(self._dev_samples),
        }

    def write_row(self, metrics: dict[str, float | int]) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "ttest_trials.csv"
        write_header = not self._header_written and not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_METRICS_COLS)
            if write_header:
                writer.writeheader()
            writer.writerow({k: metrics[k] for k in _METRICS_COLS})
        self._header_written = True
        return path

    def print_trial(self, metrics: dict[str, float | int]) -> None:
        print(
            f"  total {metrics['total_time_s']:.2f}s | "
            f"vel_x max {metrics['max_vel_x']:.2f} mean {metrics['mean_vel_x']:.2f} m/s | "
            f"path dev mean {metrics['mean_path_dev_m']:.3f} max {metrics['max_path_dev_m']:.3f} m"
        )
        print(
            f"  segments [s]: "
            f"0→1 {metrics['seg0_s']:.2f}, 1→2 {metrics['seg1_s']:.2f}, "
            f"2→3 {metrics['seg2_s']:.2f}, 3→4 {metrics['seg3_s']:.2f}, "
            f"4→5 {metrics['seg4_s']:.2f}"
        )


def _variant_prefix(env) -> str:
    n_left = len(env.unwrapped.scene["robot"].actuators["left_leg"]._joint_names)
    return "toe_equipped" if n_left == 7 else "toe_ablation"


def _resolve_resume_path(log_root_path: str, agent_cfg, checkpoint: str | None) -> str:
    """Resolve a checkpoint filename under ``log_root/<load_run>/`` (or an absolute path)."""
    if checkpoint and os.path.isfile(checkpoint):
        return os.path.abspath(checkpoint)
    run_dir = agent_cfg.load_run if agent_cfg.load_run else ".*"
    ckpt = checkpoint or agent_cfg.load_checkpoint or ".*"
    return get_checkpoint_path(log_root_path, run_dir, ckpt)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play a HyperLeg T-test policy on a fixed T-shaped waypoint path."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(str(_PROJECT_LOGS_ROOT / "rsl_rl" / agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = _resolve_resume_path(log_root_path, agent_cfg, args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    if version.parse(installed_version) < version.parse("4.0.0"):
        policy_nn = runner.alg.policy if version.parse(installed_version) >= version.parse("2.3.0") else runner.alg.actor_critic

    dt = env.unwrapped.step_dt

    controller = SequentialPoseCommandController(
        env, waypoints=_T_WAYPOINTS, arrival_std=args_cli.arrival_std,
    )
    print(
        f"[INFO] T-test play: {len(_T_WAYPOINTS)} fixed pose goals, "
        f"arrival_std={controller.cmd_term.cfg.arrival_std:.2f} m "
        f"({env.unwrapped.num_envs} env(s))."
    )

    path_nodes_rel = [(0.0, 0.0), *_T_WAYPOINTS]
    path_len = sum(
        ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        for (ax, ay), (bx, by) in zip(path_nodes_rel[:-1], path_nodes_rel[1:])
    )
    episode_s = env.unwrapped.max_episode_length * dt
    print(f"[INFO] T path length {path_len:.1f} m; episode timeout {episode_s:.1f} s.")

    draw_iface = None
    if args_cli.disable_debug_draw:
        print("[INFO] debug_draw disabled (--disable_debug_draw).")
    else:
        try:
            from isaacsim.util.debug_draw import _debug_draw  # noqa: PLC0415

            draw_iface = _debug_draw.acquire_debug_draw_interface()
        except ImportError:
            print("[WARN] isaacsim.util.debug_draw unavailable; reference path / trace disabled.")
    torso_idx = controller.robot.find_bodies("torso")[0][0]

    telemetry = None
    if args_cli.telemetry:
        try:
            from hyperleg_rl.viz import MotorTelemetryPlotter  # noqa: PLC0415
        except ImportError as exc:
            print(f"[WARN] --telemetry requires imgui_bundle. Continuing without GUI. ({exc})")
        else:
            telemetry = MotorTelemetryPlotter(env)
            telemetry.start()

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = _PROJECT_LOGS_ROOT / "Ttest" / f"{_variant_prefix(env)}_{run_stamp}"

    logger = None
    if args_cli.log_csv:
        from hyperleg_rl.viz import PowerCSVLogger  # noqa: PLC0415

        logger = PowerCSVLogger(
            env,
            log_root=_PROJECT_LOGS_ROOT / "Ttest",
            t_start_s=args_cli.log_t_start,
            t_end_s=args_cli.log_t_end,
        )
        run_dir = logger.out_dir
        print(
            f"[INFO] CSV logging armed ({logger.variant_prefix}); window "
            f"[{args_cli.log_t_start:.2f}, {args_cli.log_t_end:.2f}] s. "
            f"Out dir on first commit: {logger.out_dir}"
        )

    metrics = TTtestTrialRecorder(run_dir)
    metrics.on_trial_start(0.0)

    announced = False
    ref_drawn = False
    trial = 0
    total_times: list[float] = []
    origin_xy = env.unwrapped.scene.env_origins[0, :2]
    path_nodes_world = [
        (float(origin_xy[0] + ax), float(origin_xy[1] + ay)) for (ax, ay) in path_nodes_rel
    ]

    while simulation_app.is_running():
        start_time = time.time()
        if draw_iface is not None and not ref_drawn:
            starts = [(ax, ay, _REF_LINE_Z) for (ax, ay) in path_nodes_world[:-1]]
            ends = [(bx, by, _REF_LINE_Z) for (bx, by) in path_nodes_world[1:]]
            draw_iface.draw_lines(starts, ends, [_REF_LINE_COLOR] * len(starts), [_REF_LINE_WIDTH] * len(starts))
            ref_drawn = True

        with torch.inference_mode():
            controller.inject()
            obs = env.get_observations()
            actions = policy(obs)
            _, _, dones, _ = env.step(actions)
            controller.on_reset(dones.nonzero(as_tuple=False).flatten())
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        if telemetry is not None:
            telemetry.update()

        t_ep_s = env.unwrapped.episode_length_buf[0].item() * dt
        vel_x = controller.robot.data.root_lin_vel_b[0, 0].item()
        controller.update_path_dev()
        path_dev = controller.path_dev[0].item()
        metrics.on_step(t_ep_s, vel_x, path_dev, controller.wp_idx[0].item())

        if logger is not None:
            logger.step(t_ep_s)

        if draw_iface is not None:
            torso_xyz = controller.robot.data.body_pos_w[:, torso_idx, :]
            dev = controller.path_dev
            t_norm = (dev / max(args_cli.dev_max, 1e-6)).clamp(0.0, 1.0).tolist()
            trace_pts = [tuple(map(float, p)) for p in torso_xyz.detach().cpu().numpy()]
            colors = [_dev_to_rgba(ti) for ti in t_norm]
            draw_iface.draw_points(trace_pts, colors, [_TRACE_SIZE] * len(trace_pts))

        just_finished = bool(controller.finished[0]) and not announced
        if just_finished:
            trial += 1
            total_times.append(t_ep_s)
            row = metrics.finalize(trial, t_ep_s)
            metrics.write_row(row)
            print(f"[{trial}/{args_cli.trials}] T 경로 완주")
            metrics.print_trial(row)
            if logger is not None:
                path = logger.commit_trial(trial)
                if path is not None:
                    print(f"  power CSV → {path}")
            announced = True
            if trial >= args_cli.trials:
                break
            # Home reached — reset immediately instead of waiting for the 30 s episode timeout.
            reset_ids = torch.tensor([0], device=env.unwrapped.device)
            reset_dones = torch.zeros(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device)
            reset_dones[0] = 1
            with torch.inference_mode():
                env.reset()
                controller.on_reset(reset_ids)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(reset_dones)
                else:
                    policy_nn.reset(reset_dones)
            if logger is not None:
                logger.on_reset()
            metrics.on_trial_start(0.0)
            announced = False

        if bool(dones[0]):
            if not announced and logger is not None:
                logger.discard_trial()
            if logger is not None:
                logger.on_reset()
            metrics.on_trial_start(0.0)
            announced = False

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if total_times:
        mean_s = sum(total_times) / len(total_times)
        print(f"[done] {len(total_times)}회 평균 완주 시간 {mean_s:.2f}초 — 시뮬레이션 정지. 창을 닫으면 종료됩니다.")
        print(f"[done] trial metrics → {run_dir / 'ttest_trials.csv'}")
    if logger is not None and logger.trial_count > 0:
        print(f"[done] {logger.trial_count} power CSV(s) written under {logger.out_dir}")
        tw_paths = logger.write_torque_speed_csvs()
        if tw_paths is not None:
            print(f"[done] ankle τ-ω CSVs → {tw_paths[0].name}, {tw_paths[1].name}")

    while simulation_app.is_running():
        env.unwrapped.sim.render()
    if telemetry is not None:
        telemetry.stop()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
