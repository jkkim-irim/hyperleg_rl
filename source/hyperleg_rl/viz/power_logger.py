# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-trial steady-walking power CSV logger for HyperLeg play scripts.

Captures per-step joint-group power and the Joule-heating / mechanical-power
breakdown during a user-specified episode-time window of each trial and writes
one CSV per trial. The output reproduces the IROS/ICCAS HyperLeg paper's
Table II protocol: each row is an instantaneous sample at the 50 Hz control
rate; offline mean-aggregation of the 10 CSVs gives the Table II values.

Output layout::

    <log_root>/<variant>_<YYYY-MM-DD_HH-MM-SS>/trial_NN.csv

with ``<variant>`` ∈ {``toe_equipped``, ``toe_ablation``}, autodetected from
the actuator joint count (7 vs 6 per leg). The directory is created lazily on
the first successful commit so an aborted run leaves no empty stamped folder.

CSV columns::

    t_ep_s, joule_heating, mech_power, total_power, base_vx_w,
    l_heel_grf, r_heel_grf

All powers are in Watts. ``mech_power`` is signed (positive = motoring cost,
negative = regenerative credit). ``base_vx_w`` is the root linear velocity's
WORLD-frame x component [m/s] — the transport-CoT denominator is
``m * g * mean(base_vx_w)`` (forward progress along the goal direction,
lateral oscillation is wasted energy and is correctly NOT credited as
locomotion speed). ``l_heel_grf`` / ``r_heel_grf`` are per-foot heel
ground-reaction-force magnitudes [N] (mirrors the canonical
``heel_grf_magnitude`` obs / heel_grf_l1 reward); for steady-contact
statistics, average only over the rows where each is > 0 (zero rows are
swing phase, not "low contact"). The per-joint Joule / Mech formula mirrors
``rewards.py::_compute_cot_components`` and ``motor_telemetry.py``::

    joule  = power_coef · tau_m²                      # raw motor torque
    p      = tau_m · vel_m                             # raw motor power
    mech   = where(p < 0, p·eta_regen, p/eta_out)      # eta_out=0.9, eta_regen=0.8

Step B+: no tau asymmetric scaling — Joule is computed from raw τ_m (I²R
proxy) and driver efficiency is charged only on the mech term, matching the
paper's η_out / η_regen convention. ``rewards.py`` uses the same formula
since Step B+, so training and measurement agree.

``joule_heating`` and ``mech_power`` are summed over all motors of both legs;
``total_power = joule_heating + mech_power``. For per-joint-group breakdown
(Hip / Knee / Ankle / Toe) use the live Power tab in ``motor_telemetry.py``.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from hyperleg_rl.actuators.coupled_leg import CoupledLegActuator


# Canonical role order — mirrors CoupledLegActuator._ROLE_SUFFIXES.
_ALL_ROLES: tuple[str, ...] = ("HY", "HR", "HP", "KN", "AK", "FT", "TO")

# Per-motor electrical power coefficient Cp = R_LL / K_t^2 [W/Nm^2] by canonical role.
# SOURCE OF TRUTH: tasks/.../hyperleg/mdp/rewards.py::_POWER_COEF_BY_ROLE.
# Duplicated here so the viz package stays independent of the task package; keep
# in sync if rewards.py changes.
_POWER_COEF_BY_ROLE: dict[str, float] = {
    "HY": 3.54148162, "HR": 3.54148162, "HP": 3.54148162,
    "KN": 33.44203328, "AK": 124.567474, "FT": 124.567474, "TO": 124.567474,
}

_CSV_COLS: tuple[str, ...] = (
    "t_ep_s", "joule_heating", "mech_power", "total_power", "base_vx_w",
    "l_heel_grf", "r_heel_grf",
)


class PowerCSVLogger:
    """Buffer per-step power samples for one trial; flush on commit.

    Args:
        env: A wrapped Isaac Lab env. ``env.unwrapped.scene["robot"].actuators``
            must contain ``"left_leg"`` and ``"right_leg"`` keyed
            ``CoupledLegActuator`` instances.
        log_root: Parent directory under which the per-run folder is created.
            Typical value: ``<repo>/logs/ICCAS``.
        t_start_s: Episode-time [s] when logging opens (skip startup transient).
        t_end_s: Episode-time [s] when logging closes. Logging also closes
            early on the trial's arrival commit or on episode reset.
        env_index: Which env to log. Defaults to 0; multi-env runs log env 0 only.
        min_rows: Minimum rows required to actually write a trial CSV. Below
            this threshold, ``commit_trial`` prints a ``[WARN]`` and discards.
    """

    def __init__(
        self,
        env,
        log_root: Path,
        t_start_s: float = 2.0,
        t_end_s: float = 14.0,
        env_index: int = 0,
        min_rows: int = 50,
    ) -> None:
        robot = env.unwrapped.scene["robot"]
        self._robot = robot  # for root_lin_vel_w access in step()
        try:
            self._left: "CoupledLegActuator" = robot.actuators["left_leg"]
            self._right: "CoupledLegActuator" = robot.actuators["right_leg"]
        except KeyError as exc:
            raise RuntimeError(
                "PowerCSVLogger expects actuators keyed 'left_leg' / 'right_leg' "
                f"on robot; found {list(robot.actuators.keys())}."
            ) from exc

        n_left = len(self._left._joint_names)
        n_right = len(self._right._joint_names)
        if n_left != n_right or n_left not in (6, 7):
            raise ValueError(
                f"Unexpected per-leg joint count: L={n_left}, R={n_right} (expected 6 or 7, equal)."
            )
        self._n_roles = n_left
        self._roles = list(_ALL_ROLES[:n_left])
        # POWER_COEF in canonical role order [:n].
        self._power_coef = np.array(
            [_POWER_COEF_BY_ROLE[r] for r in self._roles], dtype=np.float64
        )

        # Heel GRF — reuse the canonical helper from the tasks package; resolve the
        # SceneEntityCfg once here, preserve_order=True pins [L, R] column order.
        # Gracefully disabled (zeros logged) if the env lacks the sensor or heels.
        self._heel = None
        try:
            from isaaclab.managers import SceneEntityCfg  # noqa: PLC0415
            from hyperleg_rl.tasks.manager_based.locomotion.velocity.hyperleg.mdp import (  # noqa: PLC0415
                heel_grf_magnitude,
            )

            heel_cfg = SceneEntityCfg(
                "contact_forces", body_names=["l_heel", "r_heel"], preserve_order=True
            )
            heel_cfg.resolve(env.unwrapped.scene)
            self._heel = {"env": env.unwrapped, "cfg": heel_cfg, "fn": heel_grf_magnitude}
        except Exception as exc:
            print(f"[PowerCSVLogger] heel GRF unavailable, columns will be zero: {exc!r}")

        self._env_index = int(env_index)
        self._t_start_s = float(t_start_s)
        self._t_end_s = float(t_end_s)
        if not (self._t_end_s > self._t_start_s >= 0.0):
            raise ValueError(
                f"Require 0 <= t_start_s < t_end_s; got ({self._t_start_s}, {self._t_end_s})."
            )
        self._min_rows = int(min_rows)

        self.variant_prefix = "toe_equipped" if n_left == 7 else "toe_ablation"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.out_dir: Path = Path(log_root) / f"{self.variant_prefix}_{timestamp}"
        self._out_dir_made = False

        self._buf: list[dict[str, float]] = []
        self._active: bool = True
        self.trial_count: int = 0

        num_envs = int(env.unwrapped.num_envs)
        if num_envs > 1:
            print(
                f"[INFO] PowerCSVLogger pinned to env {self._env_index}; "
                f"--num_envs={num_envs} but only env {self._env_index} is logged."
            )

    # ------------------------------------------------------------------
    # Lifecycle (called from the play loop).
    # ------------------------------------------------------------------
    def step(self, t_ep_s: float) -> None:
        """Append one sample if the trial is active and t_ep_s is in the window."""
        if not self._active:
            return
        if not (self._t_start_s <= t_ep_s <= self._t_end_s):
            return

        env_i = self._env_index
        # Single GPU->CPU sync per call. ``last_motor_tau`` / ``last_motor_vel`` are
        # already in canonical role order on the actuator device. ``root_lin_vel_w[0]``
        # (world +x component) and the two heel-GRF magnitudes are appended so CoT
        # uses the actual forward speed and per-foot contact loads are logged.
        chunks = [
            self._left.last_motor_tau[env_i],
            self._left.last_motor_vel[env_i],
            self._right.last_motor_tau[env_i],
            self._right.last_motor_vel[env_i],
            self._robot.data.root_lin_vel_w[env_i, 0:1],  # (vx_w,)
        ]
        if self._heel is not None:
            grf = self._heel["fn"](self._heel["env"], self._heel["cfg"])  # (E, 2)
            chunks.append(grf[env_i])  # (2,) → l_heel, r_heel
        batched = (
            torch.cat(chunks, dim=0)
            .detach()
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )
        n = self._n_roles
        tau_L = batched[0:n].astype(np.float64, copy=False)
        vel_L = batched[n:2 * n].astype(np.float64, copy=False)
        tau_R = batched[2 * n:3 * n].astype(np.float64, copy=False)
        vel_R = batched[3 * n:4 * n].astype(np.float64, copy=False)
        base_vx_w = float(batched[4 * n])
        if self._heel is not None:
            l_heel_grf = float(batched[4 * n + 1])
            r_heel_grf = float(batched[4 * n + 2])
        else:
            l_heel_grf = 0.0
            r_heel_grf = 0.0

        # Step B+: raw motor torque for Joule (I^2R proxy) and mech (τ·ω);
        # driver-side η applied only on the mech term below.
        joule_L = self._power_coef * tau_L * tau_L
        joule_R = self._power_coef * tau_R * tau_R
        pL = tau_L * vel_L
        pR = tau_R * vel_R
        # eta_regen=0.8 (paper), eta_out=0.9 (paper).
        mech_L = np.where(pL < 0.0, pL * 0.8, pL / 0.9)
        mech_R = np.where(pR < 0.0, pR * 0.8, pR / 0.9)

        joule_heating = float(joule_L.sum() + joule_R.sum())
        mech_power = float(mech_L.sum() + mech_R.sum())
        self._buf.append({
            "t_ep_s": float(t_ep_s),
            "joule_heating": joule_heating,
            "mech_power": mech_power,
            "total_power": joule_heating + mech_power,
            "base_vx_w": base_vx_w,
            "l_heel_grf": l_heel_grf,
            "r_heel_grf": r_heel_grf,
        })

    def commit_trial(self, trial_id: int) -> Path | None:
        """Write the buffered rows to ``trial_NN.csv``; return the path or None.

        Discards (no write) if fewer than ``min_rows`` rows are buffered, since
        such a thin window cannot represent steady walking.
        """
        if not self._active:
            return None
        if len(self._buf) < self._min_rows:
            print(
                f"[WARN] PowerCSVLogger: trial {trial_id:02d} has only "
                f"{len(self._buf)} rows (< {self._min_rows}); discarding."
            )
            self._buf.clear()
            self._active = False
            return None

        if not self._out_dir_made:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._out_dir_made = True

        path = self.out_dir / f"trial_{trial_id:02d}.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(_CSV_COLS))
            writer.writeheader()
            writer.writerows(self._buf)

        self._buf.clear()
        self._active = False
        self.trial_count += 1
        return path

    def discard_trial(self) -> None:
        """Drop the buffered rows (used when an episode resets without arrival)."""
        self._buf.clear()
        self._active = False

    def on_reset(self) -> None:
        """Re-arm logging for the next episode."""
        self._buf.clear()
        self._active = True
