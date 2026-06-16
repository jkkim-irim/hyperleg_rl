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
    l_grf, r_grf, joule_R_W, mech_R_W,
    hip_torque_Nm, knee_torque_Nm, ankle_torque_Nm, [toe_torque_Nm,]
    hip_power_W,  knee_power_W,  ankle_power_W,  [toe_power_W]

All powers are in Watts. ``mech_power`` is signed (positive = motoring cost,
negative = regenerative credit). ``base_vx_w`` is the root linear velocity's
WORLD-frame x component [m/s] — the transport-CoT denominator is
``m * g * mean(base_vx_w)`` (forward progress along the goal direction,
lateral oscillation is wasted energy and is correctly NOT credited as
locomotion speed). ``l_grf`` / ``r_grf`` are per-foot **heel-only** GRF magnitudes
[N]: ``||F_l_heel||`` and ``||F_r_heel||`` — identical body selection for
toe-equipped and toe-ablation (impact-absorption metrics; toe-pad force is
excluded). For steady-contact statistics, average only over rows where each
is > 0 (zero rows are swing phase). For heel-strike impact, use
``scripts/compute_heel_strike_grf.py`` (first contact sample and peak GRF in
50 ms / 100 ms post-strike windows).
The per-joint Joule / Mech formula mirrors
``rewards.py::_compute_cot_components`` and ``motor_telemetry.py``::

    joule  = power_coef · tau_m²                      # raw motor torque
    p      = tau_m · vel_m                             # raw motor power
    mech   = where(p < 0, p·eta_regen, p/eta_out)      # eta_out=0.9, eta_regen=0.8

Step B+: no tau asymmetric scaling — Joule is computed from raw τ_m (I²R
proxy) and driver efficiency is charged only on the mech term, matching the
paper's η_out / η_regen convention. ``rewards.py`` uses the same formula
since Step B+, so training and measurement agree.

``joule_heating`` and ``mech_power`` are summed over all motors of both legs;
``total_power = joule_heating + mech_power``.

Per joint-group columns are RIGHT leg only (matching the paper's single-leg
gait-cycle plots): Hip = HY+HR+HP, Knee = KN, Ankle = AK+FT, Toe = TO (toe column
present only on the 7-DoF variant). ``<grp>_power_W`` is battery power
(joule + mech) summed over that group's RIGHT-leg motors — the groups partition
the leg's motors so ``Σ <grp>_power_W == joule_R_W + mech_R_W`` (right-leg total).
Because joule is a per-motor I²R loss, this is rigorous per-actuator-group power;
it equals true per-joint power only for the decoupled hip (J = diag(25)) and is
per-actuator-group power for the CA-coupled knee/ankle/toe. ``<grp>_torque_Nm`` is
the SIGNED sum of that group's right-leg joint torques (``J^T·τ_m``, pre-friction).
``joule_R_W`` / ``mech_R_W`` are the right-leg Joule / mechanical totals (for the
single-leg power-breakdown plot). The global ``joule_heating`` / ``mech_power`` /
``total_power`` remain BOTH-legs (the full-robot power for CoT / Table II).

Additionally, the ankle torque-speed scatter behind the paper's Fig. 4 is
accumulated across all committed trials (env-0 LEFT leg) and written once at the
end via :meth:`PowerCSVLogger.write_torque_speed_csvs`::

    motor_tw.csv : L_ankle (AK) τ,ω ; R_ankle (FT) τ,ω ; knee (KN) τ,ω ;
                   [toe (TO) τ,ω when 7-DoF]
    joint_tw.csv : ankle_pitch_torque_Nm, ankle_pitch_speed_rad_s,
                   ankle_roll_torque_Nm,  ankle_roll_speed_rad_s

The motor side is the clipped per-motor τ/ω of the AK / FT differential pair;
the joint side is those signals projected through the leg Jacobian
(``τ_j = J^T·τ_m``, ``ω_j = J^{-1}·ω_m``), so Ankle-Pitch ∝ (L+R) and
Ankle-Roll ∝ (L-R), matching the Fig. 4 envelopes.
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
    "l_grf", "r_grf",
)

# Per-trial joint-group columns — RIGHT leg only, grouped by canonical motor role
# (matches the paper's single-leg gait-cycle plots). ``<grp>_power_W`` is battery
# power (joule + mech) summed over the group's RIGHT-leg motors; the groups partition
# the leg's motors so they sum to the right-leg total (= joule_R_W + mech_R_W). Power
# is a per-MOTOR quantity (joule = I²R per winding), so motor-role grouping is the
# rigorous attribution — it equals per-joint power only for the decoupled hip
# (J = diag(25)); for the CA-coupled knee/ankle/toe it is per-actuator-group power.
# ``<grp>_torque_Nm`` is the SIGNED sum of that group's right-leg joint torques
# (J^T·τ_m, pre-friction). Toe is dropped on the 6-DoF (toe-ablation) variant.
_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hip", ("HY", "HR", "HP")),
    ("knee", ("KN",)),
    ("ankle", ("AK", "FT")),
    ("toe", ("TO",)),
)

# Lower-leg motor torque-speed scatter (Fig. 4 audit). Env-0 LEFT leg, canonical
# roles AK / FT / KN / TO. Motor side = clipped per-motor τ/ω; joint side =
# τ_m @ J and ω_m @ J^{-T} for Ankle-Pitch / Ankle-Roll.
def motor_tw_cols(n_roles: int) -> tuple[str, ...]:
    """CSV header for ``motor_tw.csv`` (8 cols with toe, 6 without)."""
    cols = (
        "L_ankle_torque_Nm", "L_ankle_speed_rad_s",
        "R_ankle_torque_Nm", "R_ankle_speed_rad_s",
        "knee_torque_Nm", "knee_speed_rad_s",
    )
    if n_roles == 7:
        return cols + ("toe_torque_Nm", "toe_speed_rad_s")
    return cols


def _motor_tw_row(
    tau_L: np.ndarray,
    vel_Lc: np.ndarray,
    ak: int,
    ft: int,
    kn: int,
    to: int | None,
) -> list[float]:
    row = [
        float(tau_L[ak]), float(vel_Lc[ak]),
        float(tau_L[ft]), float(vel_Lc[ft]),
        float(tau_L[kn]), float(vel_Lc[kn]),
    ]
    if to is not None:
        row.extend([float(tau_L[to]), float(vel_Lc[to])])
    return row
_JOINT_TW_COLS: tuple[str, ...] = (
    "ankle_pitch_torque_Nm", "ankle_pitch_speed_rad_s",
    "ankle_roll_torque_Nm", "ankle_roll_speed_rad_s",
)

# Heel-only GRF (impact-absorption metric); same bodies for both robot variants.
_FOOT_GRF_LEFT = ("l_heel",)
_FOOT_GRF_RIGHT = ("r_heel",)


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

        # Ankle torque-speed capture (Fig. 4), env-0 LEFT leg only. Canonical AK / FT
        # are the differential ankle-motor pair; the leg Jacobian (cached as numpy)
        # maps the clipped motor τ/ω to the Ankle-Pitch / Ankle-Roll joint side
        # exactly as CoupledLegActuator.compute does.
        self._ak = self._roles.index("AK")
        self._ft = self._roles.index("FT")
        self._kn = self._roles.index("KN")
        self._to = self._roles.index("TO") if "TO" in self._roles else None
        self._motor_tw_cols = motor_tw_cols(n_left)
        self._J_np = self._left._J.detach().cpu().numpy().astype(np.float64)
        self._Jinv_np = self._left._J_inv.detach().cpu().numpy().astype(np.float64)

        # Joint-group columns (toe dropped on the 6-DoF variant). Indices are into
        # canonical role order, shared by the joint-torque (J^T·τ_m) and the
        # per-motor power arrays. Both legs use the same leg Jacobian.
        self._groups = [
            (name, [self._roles.index(r) for r in roles if r in self._roles])
            for name, roles in _GROUPS
        ]
        self._groups = [(name, idx) for name, idx in self._groups if idx]
        self._cols = list(_CSV_COLS)
        self._cols += ["joule_R_W", "mech_R_W"]
        self._cols += [f"{name}_torque_Nm" for name, _ in self._groups]
        self._cols += [f"{name}_power_W" for name, _ in self._groups]

        # Foot GRF — heel contact body only (||F_heel|| per side).
        self._foot = None
        try:
            from isaaclab.managers import SceneEntityCfg  # noqa: PLC0415
            from hyperleg_rl.tasks.manager_based.locomotion.velocity.hyperleg.mdp import (  # noqa: PLC0415
                foot_grf_magnitude,
            )

            left_cfg = SceneEntityCfg(
                "contact_forces", body_names=list(_FOOT_GRF_LEFT), preserve_order=True
            )
            right_cfg = SceneEntityCfg(
                "contact_forces", body_names=list(_FOOT_GRF_RIGHT), preserve_order=True
            )
            left_cfg.resolve(env.unwrapped.scene)
            right_cfg.resolve(env.unwrapped.scene)
            self._foot = {
                "env": env.unwrapped,
                "left_cfg": left_cfg,
                "right_cfg": right_cfg,
                "fn": foot_grf_magnitude,
            }
        except Exception as exc:
            print(f"[PowerCSVLogger] foot GRF unavailable, columns will be zero: {exc!r}")

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

        # Ankle τ-ω: per-trial temp buffers track _buf in lockstep; on a successful
        # commit they are folded into the persistent *_rows, written once at the end.
        self._motor_tw_buf: list[list[float]] = []
        self._joint_tw_buf: list[list[float]] = []
        self._motor_tw_rows: list[list[float]] = []
        self._joint_tw_rows: list[list[float]] = []

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
        # (world +x component) and the two per-foot GRF magnitudes are appended so CoT
        # uses the actual forward speed and per-foot contact loads are logged.
        chunks = [
            self._left.last_motor_tau[env_i],
            self._left.last_motor_vel[env_i],
            self._right.last_motor_tau[env_i],
            self._right.last_motor_vel[env_i],
            self._robot.data.root_lin_vel_w[env_i, 0:1],  # (vx_w,)
        ]
        if self._foot is not None:
            grf = self._foot["fn"](
                self._foot["env"], self._foot["left_cfg"], self._foot["right_cfg"]
            )  # (E, 2)
            chunks.append(grf[env_i])  # (2,) → l_grf, r_grf
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
        if self._foot is not None:
            l_grf = float(batched[4 * n + 1])
            r_grf = float(batched[4 * n + 2])
        else:
            l_grf = 0.0
            r_grf = 0.0

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

        # Ankle τ-ω (left leg): motor side = clipped AK / FT motor τ and ω (ω clamped
        # to ±ω_c as in CoupledLegActuator._clip_motor_4quadrant); joint side =
        # τ_m @ J and ω_m_clipped @ J^{-T} (row-batch form of J^T·τ_m, J^{-1}·ω_m).
        ak, ft = self._ak, self._ft
        wc = self._left._motor_vel_at_effort_lim[env_i].detach().cpu().numpy().astype(np.float64)
        vel_Lc = np.clip(vel_L, -wc, wc)
        jt_L = tau_L @ self._J_np
        jt_R = tau_R @ self._J_np
        joint_vel = vel_Lc @ self._Jinv_np.T
        self._motor_tw_buf.append(
            _motor_tw_row(tau_L, vel_Lc, ak, ft, self._kn, self._to)
        )
        self._joint_tw_buf.append(
            [float(jt_L[ak]), float(joint_vel[ak]), float(jt_L[ft]), float(joint_vel[ft])]
        )

        # Per-group columns — RIGHT leg only (paper single-leg gait plots). Torque =
        # SIGNED sum of the group's joint torques (J^T·τ_m); power = joule + mech over
        # the group's motors. joule_R_W / mech_R_W give the right-leg breakdown
        # (total_R = joule_R_W + mech_R_W = Σ <grp>_power_W).
        p_mot_R = joule_R + mech_R
        row = {
            "t_ep_s": float(t_ep_s),
            "joule_heating": joule_heating,
            "mech_power": mech_power,
            "total_power": joule_heating + mech_power,
            "base_vx_w": base_vx_w,
            "l_grf": l_grf,
            "r_grf": r_grf,
            "joule_R_W": float(joule_R.sum()),
            "mech_R_W": float(mech_R.sum()),
        }
        for name, idx in self._groups:
            row[f"{name}_torque_Nm"] = float(jt_R[idx].sum())
            row[f"{name}_power_W"] = float(p_mot_R[idx].sum())
        self._buf.append(row)

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
            self._motor_tw_buf.clear()
            self._joint_tw_buf.clear()
            self._active = False
            return None

        if not self._out_dir_made:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._out_dir_made = True

        path = self.out_dir / f"trial_{trial_id:02d}.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._cols)
            writer.writeheader()
            writer.writerows(self._buf)

        self._buf.clear()
        # Fold this trial's ankle τ-ω into the persistent rows, then clear the temp.
        self._motor_tw_rows.extend(self._motor_tw_buf)
        self._joint_tw_rows.extend(self._joint_tw_buf)
        self._motor_tw_buf.clear()
        self._joint_tw_buf.clear()
        self._active = False
        self.trial_count += 1
        return path

    def discard_trial(self) -> None:
        """Drop the buffered rows (used when an episode resets without arrival)."""
        self._buf.clear()
        self._motor_tw_buf.clear()
        self._joint_tw_buf.clear()
        self._active = False

    def on_reset(self) -> None:
        """Re-arm logging for the next episode."""
        self._buf.clear()
        self._motor_tw_buf.clear()
        self._joint_tw_buf.clear()
        self._active = True

    def write_torque_speed_csvs(self) -> tuple[Path, Path] | None:
        """Write accumulated ankle τ-ω (all committed trials) to two CSVs.

        ``motor_tw.csv`` holds the L / R ankle-motor torque-speed columns and
        ``joint_tw.csv`` the Ankle-Pitch / Ankle-Roll joint columns — together the
        scatter data behind Fig. 4. Returns the two paths, or None if no committed
        rows exist. Call once after the final trial.
        """
        if not self._motor_tw_rows:
            return None
        if not self._out_dir_made:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._out_dir_made = True

        motor_path = self.out_dir / "motor_tw.csv"
        joint_path = self.out_dir / "joint_tw.csv"
        for path, cols, rows in (
            (motor_path, self._motor_tw_cols, self._motor_tw_rows),
            (joint_path, _JOINT_TW_COLS, self._joint_tw_rows),
        ):
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                writer.writerows(rows)
        return motor_path, joint_path


class Fig4OperatingPointLogger:
    """Accumulate ankle τ-ω operating points from every env for Fig. 4 scatter plots.

    Each control step appends one row per env (same columns as
    :meth:`PowerCSVLogger.write_torque_speed_csvs`). Logging stops once
    ``target_rows`` samples are collected and writes::

        <log_root>/Fig4/<YYYY-MM-DD_HH-MM-SS>/motor_tw.csv
        <log_root>/Fig4/<YYYY-MM-DD_HH-MM-SS>/joint_tw.csv
    """

    def __init__(
        self,
        env,
        log_root: Path,
        target_rows: int = 20_000,
    ) -> None:
        if target_rows < 1:
            raise ValueError(f"target_rows must be >= 1; got {target_rows}.")

        robot = env.unwrapped.scene["robot"]
        left = robot.actuators["left_leg"]
        n_left = len(left._joint_names)
        if n_left not in (6, 7):
            raise ValueError(f"Unexpected per-leg joint count: {n_left} (expected 6 or 7).")

        roles = list(_ALL_ROLES[:n_left])
        self._ak = roles.index("AK")
        self._ft = roles.index("FT")
        self._kn = roles.index("KN")
        self._to = roles.index("TO") if "TO" in roles else None
        self._motor_tw_cols = motor_tw_cols(n_left)
        self._J_np = left._J.detach().cpu().numpy().astype(np.float64)
        self._Jinv_np = left._J_inv.detach().cpu().numpy().astype(np.float64)
        self._motor_vel_at_effort_lim = (
            left._motor_vel_at_effort_lim[0].detach().cpu().numpy().astype(np.float64)
        )

        self._left = left
        self._target_rows = int(target_rows)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.out_dir: Path = Path(log_root) / "Fig4" / timestamp
        self._motor_tw_rows: list[list[float]] = []
        self._joint_tw_rows: list[list[float]] = []

        num_envs = int(env.unwrapped.num_envs)
        print(
            f"[INFO] Fig4OperatingPointLogger: target {self._target_rows} rows "
            f"from {num_envs} env(s) → {self.out_dir}"
        )

    @property
    def row_count(self) -> int:
        return len(self._motor_tw_rows)

    @property
    def is_full(self) -> bool:
        return self.row_count >= self._target_rows

    def step(self) -> bool:
        """Append up to one row per env. Returns True when ``target_rows`` is reached."""
        if self.is_full:
            return True

        ak, ft = self._ak, self._ft
        tau_L = self._left.last_motor_tau.detach().cpu().numpy().astype(np.float64)
        vel_L = self._left.last_motor_vel.detach().cpu().numpy().astype(np.float64)
        vel_Lc = np.clip(vel_L, -self._motor_vel_at_effort_lim, self._motor_vel_at_effort_lim)
        jt_L = tau_L @ self._J_np
        joint_vel = vel_Lc @ self._Jinv_np.T

        motor_rows = [
            _motor_tw_row(tau_L[i], vel_Lc[i], ak, ft, self._kn, self._to)
            for i in range(tau_L.shape[0])
        ]
        motor_batch = np.asarray(motor_rows, dtype=np.float64)
        joint_batch = np.stack(
            (jt_L[:, ak], joint_vel[:, ak], jt_L[:, ft], joint_vel[:, ft]), axis=1
        )

        remaining = self._target_rows - self.row_count
        take = min(remaining, motor_batch.shape[0])
        self._motor_tw_rows.extend(motor_batch[:take].tolist())
        self._joint_tw_rows.extend(joint_batch[:take].tolist())
        return self.is_full

    def write_csvs(self) -> tuple[Path, Path]:
        """Write ``motor_tw.csv`` and ``joint_tw.csv`` under :attr:`out_dir`."""
        if not self._motor_tw_rows:
            raise RuntimeError("No Fig. 4 operating points collected.")

        motor_rows = self._motor_tw_rows[: self._target_rows]
        joint_rows = self._joint_tw_rows[: self._target_rows]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        motor_path = self.out_dir / "motor_tw.csv"
        joint_path = self.out_dir / "joint_tw.csv"
        for path, cols, rows in (
            (motor_path, self._motor_tw_cols, motor_rows),
            (joint_path, _JOINT_TW_COLS, joint_rows),
        ):
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                writer.writerows(rows)
        return motor_path, joint_path
