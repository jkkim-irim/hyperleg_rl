# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive a trained HyperLeg velocity policy to a world-frame +x goal in a straight line.

The policy was trained to track a *base-frame* velocity command ``[vx, vy, wz]``.
Commanding a constant base-frame ``vx`` makes the robot drift sideways in the world
frame whenever the base yaw wanders. This script closes that loop: every control
step it overrides the velocity command with a small unicycle controller that

  * drives forward (base ``vx``) at a constant speed, then a single hard switch to 0
    once the goal x is reached (bang-bang — no ramp, so the robot's own braking and
    overshoot, e.g. with vs without toes, is observable rather than masked), and
  * steers (base ``wz``) to cancel world-frame cross-track (y) drift,

so the robot walks straight to ``world x = env_origin_x + goal_x`` and stops there.
No retraining — the trained policy is used as-is.

Mechanism (verified against IsaacLab source): inside ``env.step()`` the command
manager recomputes ``vel_command_b`` right before observations, so we cannot rely
on writing the buffer and stepping. Instead, each iteration we override the buffer
from the *current* robot state and then call ``env.get_observations()`` (which
recomputes the policy obs fresh) before inference. The observation returned by
``env.step()`` is discarded.

Example:
    ./isaaclab.sh -p scripts/rsl_rl/play_goto_x.py \
        --task HyperLeg-Locomotion-Play-v0 --load_run 2026-05-29_13-22-53 \
        --goal_x 20.0 --num_envs 1 --real-time
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Walk a trained HyperLeg policy to a world +x goal.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="HyperLeg-Locomotion-Play-v0", help="Name of the task.")
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
# -- goal / controller arguments
parser.add_argument("--goal_x", type=float, default=20.0, help="Forward (+x) distance to walk, in meters.")
parser.add_argument("--speed", type=float, default=1.33, help="Constant forward speed [m/s] held until the goal, then hard-stopped.")
parser.add_argument("--dev_max", type=float, default=0.5, help="Path deviation [m] that maps the trace color to full red.")
parser.add_argument("--disable_debug_draw", action="store_true", default=False, help="Turn off the debug_draw reference line and trajectory trace.")
parser.add_argument("--trials", type=int, default=10, help="Number of arrival trials to log, then exit.")
# -- steering gains (defaults tuned for firm line-holding; raise further to hold harder)
parser.add_argument("--k_ct", type=float, default=2.5, help="Cross-track gain: world-y error -> desired heading [rad/m].")
parser.add_argument("--kp_yaw", type=float, default=1.5, help="Heading-error gain -> yaw-rate command [1/s].")
parser.add_argument("--max_heading", type=float, default=0.4, help="Cap on desired heading toward the line [rad].")
parser.add_argument("--wz_max", type=float, default=1.0, help="Yaw-rate command cap [rad/s] (keep <= 1.0 for training range).")
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
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
import hyperleg_rl.tasks  # noqa: F401 — register hyperleg gym envs
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Project-local logs root: <hyperleg_rl repo>/logs
_PROJECT_LOGS_ROOT = Path(__file__).resolve().parents[2] / "logs"

# Reference-path debug line, world frame: (0, 0, z) -> (goal_x, 0, z). 연두색 (yellow-green).
_REF_LINE_Z = 1.0
_REF_LINE_COLOR = (0.6, 1.0, 0.2, 1.0)  # RGBA
_REF_LINE_WIDTH = 5.0  # pixels

# Trajectory trace: a point at the /torso world position every control step (50 Hz).
# Color encodes deviation from the reference line (0,0)->(goal_x,0): mint -> green ->
# yellow -> orange -> red as the torso strays further off the path (full red at dev_max).
_TRACE_SIZE = 6.0  # pixels
_DEV_CMAP = (
    (0.00, (0.40, 1.00, 0.80)),  # mint (on the line)
    (0.25, (0.10, 0.90, 0.10)),  # green
    (0.50, (1.00, 1.00, 0.00)),  # yellow
    (0.75, (1.00, 0.55, 0.00)),  # orange
    (1.00, (1.00, 0.00, 0.00)),  # red (>= dev_max off the line)
)


def _dev_to_rgba(t: float) -> tuple:
    """Map a normalized deviation ``t`` in [0, 1] to an RGBA color along ``_DEV_CMAP``."""
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    for (t0, c0), (t1, c1) in zip(_DEV_CMAP, _DEV_CMAP[1:]):
        if t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return (c0[0] + f * (c1[0] - c0[0]), c0[1] + f * (c1[1] - c0[1]), c0[2] + f * (c1[2] - c0[2]), 1.0)
    return (*_DEV_CMAP[-1][1], 1.0)


class WorldXGoalController:
    """Unicycle controller that overrides the base-frame velocity command each step.

    Walks the robot to ``goal = env_origin + (goal_x, 0)`` in world frame, keeping a
    straight line along ``y = env_origin_y`` by steering against cross-track drift.
    Forward speed is bang-bang: held at ``speed`` until the goal x is reached, then a
    single hard switch to 0 (no ramp) so the robot's own braking / overshoot shows.
    All quantities are batched over envs.
    """

    def __init__(self, env, goal_x: float, speed: float, k_ct: float = 2.5,
                 kp_yaw: float = 1.5, max_heading: float = 0.4, wz_max: float = 1.0):
        self.robot = env.unwrapped.scene["robot"]
        self.cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        device = env.unwrapped.device
        # Make this controller the single source of truth for the command: neutralize
        # the built-in UniformVelocityCommand so env.step()'s command_manager.compute()
        # no longer resamples or applies heading control to vel_command_b[:, 2]. This is
        # purely local to the live term instance — the shared env cfg is untouched.
        self.cmd_term.cfg.heading_command = False
        self.cmd_term.cfg.rel_standing_envs = 0.0
        self.cmd_term.is_standing_env[:] = False
        # goal in world frame: each env's origin shifted by +goal_x in x
        origins = env.unwrapped.scene.env_origins  # (N, 3), absolute world
        self.goal_xy = origins[:, :2].clone()
        self.goal_xy[:, 0] += goal_x
        self.speed = float(speed)
        # Steering gains / limits (tunable via CLI). Raising k_ct / kp_yaw holds the line
        # harder; keep wz_max <= 1.0 to stay inside the trained ang_vel_z range, and note
        # that too-stiff gains can make the path weave (policy lag at 50 Hz).
        self.K_CT = float(k_ct)
        self.MAX_HEADING = float(max_heading)
        self.KP_YAW = float(kp_yaw)
        self.WZ_MAX = float(wz_max)
        self.arrived = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=device)

    def on_reset(self, done_ids: torch.Tensor) -> None:
        """Clear the arrival latch for envs that were just reset (robot back at origin)."""
        if done_ids.numel() > 0:
            self.arrived[done_ids] = False

    def inject(self) -> torch.Tensor:
        """Compute the command from the current robot state and write it into the buffer.

        Returns the per-env remaining +x distance.
        """
        pos = self.robot.data.root_pos_w[:, :2]  # (N, 2) world x, y
        yaw = self.robot.data.heading_w  # (N,) world yaw in [-pi, pi]

        dx = self.goal_xy[:, 0] - pos[:, 0]  # remaining +x distance
        cross = pos[:, 1] - self.goal_xy[:, 1]  # +ve = drifted in +y

        # Desired heading steers back toward the line (->0 = face world +x), capped.
        heading_des = (-self.K_CT * cross).clamp(-self.MAX_HEADING, self.MAX_HEADING)
        e_yaw = math_utils.wrap_to_pi(heading_des - yaw)
        wz = (self.KP_YAW * e_yaw).clamp(-self.WZ_MAX, self.WZ_MAX)

        # Bang-bang forward: hold a constant speed until the goal x is reached, then a
        # single hard switch to 0 (no deceleration ramp). The robot's own braking and
        # overshoot — and how it differs with vs without toes — is then visible rather
        # than masked by a velocity profile.
        self.arrived |= dx <= 0.0
        v = torch.where(self.arrived, torch.zeros_like(dx), torch.full_like(dx, self.speed))
        wz = torch.where(self.arrived, torch.zeros_like(wz), wz)

        # Base-frame command [vx, vy, wz]; lateral velocity held at zero.
        cmd = torch.stack([v, torch.zeros_like(v), wz], dim=-1)
        self.cmd_term.vel_command_b[:] = cmd
        return dx


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play a HyperLeg policy with a world +x goal controller."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # resolve checkpoint (mirrors play.py)
    log_root_path = os.path.abspath(str(_PROJECT_LOGS_ROOT / "rsl_rl" / agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    if version.parse(installed_version) < version.parse("4.0.0"):
        policy_nn = runner.alg.policy if version.parse(installed_version) >= version.parse("2.3.0") else runner.alg.actor_critic

    dt = env.unwrapped.step_dt

    controller = WorldXGoalController(
        env, goal_x=args_cli.goal_x, speed=args_cli.speed,
        k_ct=args_cli.k_ct, kp_yaw=args_cli.kp_yaw, max_heading=args_cli.max_heading, wz_max=args_cli.wz_max,
    )
    print(
        f"[INFO] Walking to world +x goal: +{args_cli.goal_x:.2f} m at up to {args_cli.speed:.2f} m/s "
        f"({env.unwrapped.num_envs} env(s))."
    )
    # Warn if the goal cannot plausibly be reached before the episode times out and resets.
    episode_s = env.unwrapped.max_episode_length * dt
    walk_budget_s = args_cli.goal_x / max(args_cli.speed, 1e-6)
    if walk_budget_s > 0.8 * episode_s:
        print(
            f"[WARN] ~{walk_budget_s:.1f}s of walking needed but the episode resets at {episode_s:.1f}s; "
            f"the robot may loop without reaching the goal. Lower --goal_x or raise --speed."
        )

    # Acquire the debug-draw interface for the reference line + trajectory trace (same
    # API as the feet-force viz). Skipped entirely when --disable_debug_draw is set, or
    # gracefully if unavailable (e.g. headless). draw_iface stays None => every draw and
    # the per-step deviation math below is short-circuited.
    draw_iface = None
    if args_cli.disable_debug_draw:
        print("[INFO] debug_draw disabled (--disable_debug_draw).")
    else:
        try:
            from isaacsim.util.debug_draw import _debug_draw  # noqa: PLC0415

            draw_iface = _debug_draw.acquire_debug_draw_interface()
        except ImportError:
            print("[WARN] isaacsim.util.debug_draw unavailable; reference line / trace disabled.")
    torso_idx = controller.robot.find_bodies("torso")[0][0]  # body index for the trajectory trace

    # optional live telemetry plotter (env-0 only) — same as play.py
    telemetry = None
    if args_cli.telemetry:
        try:
            from hyperleg_rl.viz import MotorTelemetryPlotter  # noqa: PLC0415
        except ImportError as exc:
            print(f"[WARN] --telemetry requires imgui_bundle. Continuing without GUI. ({exc})")
        else:
            telemetry = MotorTelemetryPlotter(env)
            telemetry.start()

    announced = False  # arrival already logged this episode (env 0)
    ref_drawn = False
    trial = 0
    arrival_times = []
    while simulation_app.is_running():
        start_time = time.time()
        # Draw the world-frame reference line once, during an active sim frame. It is
        # static, so a single draw_lines() call persists until clear_lines() (never).
        if draw_iface is not None and not ref_drawn:
            draw_iface.draw_lines(
                [(0.0, 0.0, _REF_LINE_Z)],
                [(float(args_cli.goal_x), 0.0, _REF_LINE_Z)],
                [_REF_LINE_COLOR],
                [_REF_LINE_WIDTH],
            )
            ref_drawn = True
        # Override the command from the *current* state, then recompute obs so the
        # policy sees our command (env.step's own command/obs are discarded).
        controller.inject()
        with torch.inference_mode():
            obs = env.get_observations()
            actions = policy(obs)
            _, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        controller.on_reset(dones.nonzero(as_tuple=False).flatten())
        if telemetry is not None:
            telemetry.update()

        # Trajectory trace: drop a point at the torso world position each control step
        # (50 Hz). Color encodes the torso's distance from the reference segment
        # (0,0)->(goal_x,0): inside the x-range that distance is just |y| (cross-track),
        # past the endpoint it also picks up the x overshoot. Points are never cleared,
        # so every trial's walk accumulates and stays visible (the teleport on reset just
        # leaves a gap between point clusters — no false connecting line, since these are points).
        if draw_iface is not None:
            torso_xyz = controller.robot.data.body_pos_w[:, torso_idx, :]  # (N, 3) world
            gx = float(args_cli.goal_x)
            dy = torso_xyz[:, 1]  # cross-track from line y = 0 (the whole error while x <= goal_x)
            dx_over = torch.clamp(torso_xyz[:, 0] - gx, min=0.0)  # 0 until x passes goal_x, then the overshoot
            dev = torch.sqrt(dx_over * dx_over + dy * dy)  # single formula; x only enters past the goal [m]
            t_norm = (dev / max(args_cli.dev_max, 1e-6)).clamp(0.0, 1.0).tolist()
            trace_pts = [tuple(map(float, p)) for p in torso_xyz.detach().cpu().numpy()]
            colors = [_dev_to_rgba(ti) for ti in t_norm]
            draw_iface.draw_points(trace_pts, colors, [_TRACE_SIZE] * len(trace_pts))

        # Log one line per trial: the sim time taken to reach the goal, measured from the
        # episode start (env 0). Re-arm on reset; exit after the requested trial count.
        if bool(dones[0]):
            announced = False
        elif bool(controller.arrived[0]) and not announced:
            arrival_s = env.unwrapped.episode_length_buf[0].item() * dt
            trial += 1
            arrival_times.append(arrival_s)
            print(f"[{trial}/{args_cli.trials}] 도달 시간 {arrival_s:.2f}초")
            announced = True
            if trial >= args_cli.trials:
                break

        # real-time pacing
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if arrival_times:
        mean_s = sum(arrival_times) / len(arrival_times)
        print(f"[done] {len(arrival_times)}회 평균 도달 시간 {mean_s:.2f}초 — 시뮬레이션 정지. 창을 닫으면 종료됩니다.")
    # Stop stepping the policy/physics but keep the viewer open with every trajectory
    # retained: render in place (no env.step => the robot is frozen) until the window closes.
    while simulation_app.is_running():
        env.unwrapped.sim.render()
    if telemetry is not None:
        telemetry.stop()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
