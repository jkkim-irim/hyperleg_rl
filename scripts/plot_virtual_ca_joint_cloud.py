# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Virtual CA operating points: sample motor envelopes → map to joint via J.

For each of ``n_points`` samples, draw (τ, ω) independently inside the DC
motor envelope of KN / AK (ankle_L) / FT (ankle_R) / TO, then map to joint
space with the same linear rules as :class:`CoupledLegActuator` logging::

    τ_joint = τ_motor @ J
    ω_joint = ω_motor @ J^{-T}

With ``--grid``, fills each motor envelope on a regular τ–ω lattice (MATLAB-style),
writes four motor-level PNGs, then joint scatter panels.

Run::

    python scripts/plot_virtual_ca_joint_cloud.py --grid --grid_side 100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import Rectangle

from plot_fig04_ca_envelopes import (
    AK,
    BLUE_Q,
    ENV_LS,
    FT,
    GREEN_FILL,
    GREEN_LINE,
    J,
    KN,
    PURPLE_FILL,
    PURPLE_LINE,
    RED_Q,
    TO,
    _ANKLE_MOTOR_XLIM,
    _ANKLE_MOTOR_YLIM,
    _ANKLE_MOTOR_YTICKS,
    _KNEE_MOTOR_XLIM,
    _KNEE_MOTOR_YLIM,
    _KNEE_MOTOR_YTICKS,
    _motor_limits,
    joint_envelope,
    motor_envelope,
)

_MOTOR_PANELS: tuple[tuple[str, int, str, tuple, tuple, list, list], ...] = (
    ("Knee motor (KN)", KN, "fig_virtual_motor_KN",
     _KNEE_MOTOR_XLIM, _KNEE_MOTOR_YLIM, [-2, 0, 2], _KNEE_MOTOR_YTICKS),
    ("Ankle Pitch motor (AK)", AK, "fig_virtual_motor_AK",
     _ANKLE_MOTOR_XLIM, _ANKLE_MOTOR_YLIM, [-1, 0, 1], _ANKLE_MOTOR_YTICKS),
    ("Ankle Roll motor (FT)", FT, "fig_virtual_motor_FT",
     _ANKLE_MOTOR_XLIM, _ANKLE_MOTOR_YLIM, [-1, 0, 1], _ANKLE_MOTOR_YTICKS),
    ("Toe motor (TO)", TO, "fig_virtual_motor_TO",
     _ANKLE_MOTOR_XLIM, _ANKLE_MOTOR_YLIM, [-1, 0, 1], _ANKLE_MOTOR_YTICKS),
)

_LOWER_LEG_MOTORS: tuple[tuple[str, int], ...] = (
    ("KN", KN),
    ("AK", AK),
    ("FT", FT),
    ("TO", TO),
)

_JOINT_PANELS: tuple[tuple[str, int, str], ...] = (
    ("Knee", KN, "fig_virtual_joint_Knee"),
    ("Ankle Pitch", AK, "fig_virtual_joint_AnklePitch"),
    ("Ankle Roll", FT, "fig_virtual_joint_AnkleRoll"),
    ("Toe", TO, "fig_virtual_joint_Toe"),
)

_SCATTER = "#1565c0"


def _motor_envelope_path(motor_idx: int, n: int = 400) -> MplPath:
    tx, ty = motor_envelope(motor_idx, n=n)
    return MplPath(np.column_stack([tx, ty]))


def _sample_motor_envelope(motor_idx: int, rng: np.random.Generator) -> tuple[float, float]:
    """Uniform sample inside one motor DC τ–ω envelope (rejection)."""
    t_s, _, wc = _motor_limits(motor_idx)
    path = _motor_envelope_path(motor_idx)
    tau_lo, tau_hi = -t_s, t_s
    omega_lo, omega_hi = -wc, wc
    while True:
        tau = rng.uniform(tau_lo, tau_hi)
        omega = rng.uniform(omega_lo, omega_hi)
        if path.contains_point((tau, omega)):
            return float(tau), float(omega)


def _sample_motor_grid(motor_idx: int, n_tau: int, n_omega: int) -> np.ndarray:
    """Regular (τ, ω) grid clipped to the motor envelope (MATLAB-style fill)."""
    t_s, _, wc = _motor_limits(motor_idx)
    path = _motor_envelope_path(motor_idx)
    taus = np.linspace(-t_s, t_s, n_tau)
    omegas = np.linspace(-wc, wc, n_omega)
    tt, oo = np.meshgrid(taus, omegas)
    pts = np.column_stack([tt.ravel(), oo.ravel()])
    return pts[path.contains_points(pts)]


def _sample_motor_interior(motor_idx: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform samples filling the DC motor envelope interior (rejection, batched)."""
    t_s, _, wc = _motor_limits(motor_idx)
    path = _motor_envelope_path(motor_idx)
    out = np.empty((n, 2), dtype=np.float64)
    filled = 0
    batch = 16_384
    while filled < n:
        tau = rng.uniform(-t_s, t_s, batch)
        omega = rng.uniform(-wc, wc, batch)
        pts = np.column_stack([tau, omega])
        keep = pts[path.contains_points(pts)]
        take = min(keep.shape[0], n - filled)
        if take:
            out[filled : filled + take] = keep[:take]
            filled += take
    return out


def _sample_motor_envelope_dense(motor_idx: int, n: int = 2500) -> np.ndarray:
    """Dense samples on the DC τ–ω envelope boundary (``2·n`` points, line-like)."""
    tx, ty = motor_envelope(motor_idx, n=n)
    return np.column_stack([tx, ty])


def _filter_grid_to_envelope_boundary(pts: np.ndarray, motor_idx: int) -> np.ndarray:
    """Keep lattice points on the DC envelope boundary; drop interior grid points."""
    path = _motor_envelope_path(motor_idx)
    taus = np.unique(pts[:, 0])
    omegas = np.unique(pts[:, 1])
    boundary: list[list[float]] = []
    for tau, omega in pts:
        if not path.contains_point((tau, omega)):
            continue
        ti = int(np.argmin(np.abs(taus - tau)))
        oj = int(np.argmin(np.abs(omegas - omega)))
        on_edge = False
        if ti == 0 or not path.contains_point((taus[ti - 1], omega)):
            on_edge = True
        elif ti == len(taus) - 1 or not path.contains_point((taus[ti + 1], omega)):
            on_edge = True
        elif oj == 0 or not path.contains_point((tau, omegas[oj - 1])):
            on_edge = True
        elif oj == len(omegas) - 1 or not path.contains_point((tau, omegas[oj + 1])):
            on_edge = True
        if on_edge:
            boundary.append([float(tau), float(omega)])
    return np.asarray(boundary, dtype=np.float64)


def _sample_motor_grid_boundary(motor_idx: int, n_tau: int, n_omega: int) -> np.ndarray:
    """Dense DC envelope boundary (replaces sparse lattice boundary)."""
    return _sample_motor_envelope_dense(motor_idx, n=max(n_tau, n_omega) * 25)


def sample_motor_states(
    n_points: int,
    rng: np.random.Generator,
    *,
    grid: bool,
    grid_side: int,
    per_motor_grids: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(tau_m, omega_m)`` with shape ``(n_points, 7)``; hips are zero."""
    tau_m = np.zeros((n_points, 7), dtype=np.float64)
    omega_m = np.zeros((n_points, 7), dtype=np.float64)

    if grid:
        grids = per_motor_grids or [
            _sample_motor_grid(midx, grid_side, grid_side) for _, midx in _LOWER_LEG_MOTORS
        ]
        for i in range(n_points):
            for j, (_, midx) in enumerate(_LOWER_LEG_MOTORS):
                pt = grids[j][rng.integers(grids[j].shape[0])]
                tau_m[i, midx] = pt[0]
                omega_m[i, midx] = pt[1]
        return tau_m, omega_m

    for i in range(n_points):
        for _, midx in _LOWER_LEG_MOTORS:
            tau_m[i, midx], omega_m[i, midx] = _sample_motor_envelope(midx, rng)
    return tau_m, omega_m


def _panel_motor_grid(
    ax,
    motor_idx: int,
    tau: np.ndarray,
    omega: np.ndarray,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xticks: list[float],
    yticks: list[float],
) -> None:
    """Motor τ–ω panel: DC envelope + grid-filled virtual operating points."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    xmin, xmax = xlim
    ymin, ymax = ylim
    for (x0, x1, y0, y1), c in (
        ((0, xmax, 0, ymax), RED_Q), ((xmin, 0, ymin, 0), RED_Q),
        ((xmin, 0, 0, ymax), BLUE_Q), ((0, xmax, ymin, 0), BLUE_Q),
    ):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=c, edgecolor="none", zorder=0))

    ax.grid(True, color="0.85", lw=0.7, zorder=0.5)
    ax.axhline(0.0, ls="--", color="0.5", lw=1.0, zorder=0.6)
    ax.axvline(0.0, ls="--", color="0.5", lw=1.0, zorder=0.6)

    ex, ey = motor_envelope(motor_idx)
    ax.fill(ex, ey, facecolor=GREEN_FILL, alpha=0.35, zorder=1)
    ax.plot(
        np.append(ex, ex[0]), np.append(ey, ey[0]),
        color=GREEN_LINE, lw=1.0, ls=ENV_LS, zorder=3, label="DC motor envelope",
    )
    ax.scatter(
        tau, omega,
        s=1 if len(tau) > 5_000 else (2 if len(tau) > 800 else 6),
        c=_SCATTER,
        alpha=0.12 if len(tau) > 5_000 else (0.85 if len(tau) > 800 else 0.55),
        edgecolors="none",
        zorder=2,
        rasterized=len(tau) > 5_000,
        label="Virtual op. points",
    )

    tag = dict(color="0.45", fontsize=12, fontweight="bold", zorder=4)
    pad_x, pad_y = 0.04 * (xmax - xmin), 0.05 * (ymax - ymin)
    ax.text(xmin + pad_x, ymax - pad_y, "Q2", ha="left", va="top", **tag)
    ax.text(xmin + pad_x, ymin + pad_y, "Q3", ha="left", va="bottom", **tag)
    ax.text(xmax - pad_x, ymin + pad_y, "Q4", ha="right", va="bottom", **tag)

    ax.set_title(title)
    ax.set_xlabel("Torque [Nm]")
    ax.set_ylabel("Speed [rad/s]")
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)


def _save_motor_grid_figures(
    out_dir: Path,
    grids: list[np.ndarray],
    *,
    dpi: int,
    title_tag: str = "grid samples",
) -> None:
    for (label, midx, stem, xlim, ylim, xticks, yticks), pts in zip(_MOTOR_PANELS, grids):
        tau, omega = pts[:, 0], pts[:, 1]
        fig, ax = plt.subplots(figsize=(5.0, 4.4))
        _panel_motor_grid(
            ax, midx, tau, omega,
            f"{label} — {title_tag} (N={len(tau)})",
            xlim, ylim, xticks, yticks,
        )
        fig.tight_layout()
        out = out_dir / f"{stem}.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(
            f"[saved] {out}  τ∈[{tau.min():.2f},{tau.max():.2f}] "
            f"ω∈[{omega.min():.0f},{omega.max():.0f}]  (N={len(tau)})"
        )


def _motor_states_aligned(per_motor_grids: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pair motor CSV rows by index: row *i* from each KN/AK/FT/TO → one joint sample."""
    n = min(g.shape[0] for g in per_motor_grids)
    tau_m = np.zeros((n, 7), dtype=np.float64)
    omega_m = np.zeros((n, 7), dtype=np.float64)
    for j, (_, midx) in enumerate(_LOWER_LEG_MOTORS):
        tau_m[:, midx] = per_motor_grids[j][:n, 0]
        omega_m[:, midx] = per_motor_grids[j][:n, 1]
    return tau_m, omega_m


def motor_to_joint(tau_m: np.ndarray, omega_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map batched motor (τ, ω) to joint (τ, ω) via ``hyperleg.py`` Jacobian."""
    j_inv_t = np.linalg.inv(J).T
    tau_j = tau_m @ J
    omega_j = omega_m @ j_inv_t
    return tau_j, omega_j


_ENVELOPE_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _cached_joint_envelope(col: int) -> tuple[np.ndarray, np.ndarray]:
    if col not in _ENVELOPE_CACHE:
        _ENVELOPE_CACHE[col] = joint_envelope(col, n_motor=120)
    return _ENVELOPE_CACHE[col]


def _panel_joint_cloud(
    ax,
    tau: np.ndarray,
    omega: np.ndarray,
    col: int,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    show_envelope: bool,
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    xmin, xmax = xlim
    ymin, ymax = ylim
    for (x0, x1, y0, y1), c in (
        ((0, xmax, 0, ymax), RED_Q), ((xmin, 0, ymin, 0), RED_Q),
        ((xmin, 0, 0, ymax), BLUE_Q), ((0, xmax, ymin, 0), BLUE_Q),
    ):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=c, edgecolor="none", zorder=0))

    ax.grid(True, color="0.85", lw=0.7, zorder=0.5)
    ax.axhline(0.0, ls="--", color="0.5", lw=1.0, zorder=0.6)
    ax.axvline(0.0, ls="--", color="0.5", lw=1.0, zorder=0.6)

    if show_envelope:
        ex, ey = _cached_joint_envelope(col)
        ax.fill(ex, ey, facecolor=PURPLE_FILL, alpha=0.12, zorder=1)
        ax.plot(
            np.append(ex, ex[0]), np.append(ey, ey[0]),
            color=PURPLE_LINE, lw=1.5, ls="-", zorder=4,
            label="Joint envelope (Minkowski)",
        )

    n = len(tau)
    ax.scatter(
        tau, omega,
        s=4 if n > 10_000 else 8,
        c=_SCATTER,
        alpha=0.45 if n > 10_000 else 0.7,
        edgecolors="none",
        zorder=2,
        rasterized=n > 10_000,
        label="Virtual op. points",
    )
    ax.set_title(title)
    ax.set_xlabel("Torque [Nm]")
    ax.set_ylabel("Speed [rad/s]")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8)


def _load_motor_grids_from_csv(out_dir: Path) -> list[np.ndarray]:
    grids: list[np.ndarray] = []
    for _, _, stem, *_ in _MOTOR_PANELS:
        path = out_dir / f"{stem}.csv"
        with path.open() as f:
            rows = list(csv.reader(f))[1:]
        grids.append(np.array([[float(v) for v in r] for r in rows], dtype=np.float64))
    return grids


def _load_joint_tw_csv(path: Path) -> np.ndarray:
    with path.open() as f:
        rows = list(csv.reader(f))[1:]
    return np.array([[float(v) for v in r] for r in rows], dtype=np.float64)


def _save_joint_figures(
    joint: np.ndarray,
    out_dir: Path,
    *,
    dpi: int,
    show_envelope: bool,
) -> None:
    n = joint.shape[0]
    cols = (KN, AK, FT, TO)
    for (label, col, stem), jcol in zip(_JOINT_PANELS, range(4)):
        tau = joint[:, jcol * 2]
        omega = joint[:, jcol * 2 + 1]
        col_idx = cols[jcol]
        xlim, ylim = _joint_axis_limits(tau, omega, col_idx, show_envelope=show_envelope)
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        _panel_joint_cloud(
            ax, tau, omega, col_idx, f"{label} — virtual motor samples (N={n})",
            xlim, ylim, show_envelope=show_envelope,
        )
        fig.tight_layout()
        out = out_dir / f"{stem}.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        ex, ey = _cached_joint_envelope(cols[jcol])
        print(
            f"[saved] {out}  scatter τ∈[{tau.min():.1f},{tau.max():.1f}] "
            f"ω∈[{omega.min():.1f},{omega.max():.1f}]  "
            f"envelope τ∈[{ex.min():.1f},{ex.max():.1f}] ω∈[{ey.min():.1f},{ey.max():.1f}]"
        )


def _auto_limits(vals: np.ndarray, pad: float = 0.06) -> tuple[float, float]:
    lo, hi = float(vals.min()), float(vals.max())
    span = hi - lo
    margin = pad * span if span > 0 else 1.0
    return lo - margin, hi + margin


def _joint_axis_limits(
    tau: np.ndarray,
    omega: np.ndarray,
    col: int,
    *,
    show_envelope: bool,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if not show_envelope:
        return _auto_limits(tau), _auto_limits(omega)
    ex, ey = _cached_joint_envelope(col)
    return _auto_limits(np.concatenate([tau, ex])), _auto_limits(np.concatenate([omega, ey]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--n_points", type=int, default=10_000, help="Number of virtual samples.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Grid-fill each motor envelope; also write four motor-level PNGs.",
    )
    parser.add_argument(
        "--boundary_only",
        action="store_true",
        help="With --grid: keep envelope boundary lattice points only (no interior fill).",
    )
    parser.add_argument(
        "--grid_side",
        type=int,
        default=100,
        help="τ and ω lattice resolution per motor when --grid (default 100).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=repo / "ICCAS/figures/virtual_joint_cloud",
        help="Output directory for PNG / CSV.",
    )
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--no_envelope", action="store_true", help="Omit Minkowski envelope overlay.")
    parser.add_argument(
        "--joint_only",
        action="store_true",
        help="Map saved motor grid CSVs → joint PNGs only (skip motor figures).",
    )
    parser.add_argument(
        "--filter_boundary",
        action="store_true",
        help="Rewrite motor CSV/PNG with dense DC-envelope boundary samples.",
    )
    parser.add_argument(
        "--boundary_samples",
        type=int,
        default=2500,
        help="Half-envelope resolution; total boundary points ≈ 2× this (default 2500→5000).",
    )
    parser.add_argument(
        "--interior_fill",
        action="store_true",
        help="Rewrite motor CSV/PNG with uniform interior envelope fill.",
    )
    parser.add_argument(
        "--motor_points",
        type=int,
        default=50_000,
        help="Points per motor for --interior_fill (default 50000).",
    )
    parser.add_argument(
        "--from_csv",
        action="store_true",
        help="Replot joint PNGs from existing virtual_joint_tw.csv (no resample).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        csv_path = args.out_dir / "virtual_joint_tw.csv"
        joint = _load_joint_tw_csv(csv_path)
        _save_joint_figures(
            joint, args.out_dir, dpi=min(args.dpi, 200), show_envelope=not args.no_envelope,
        )
        print(f"[replotted] {csv_path}  ({joint.shape[0]} rows)")
        return

    if args.interior_fill:
        rng = np.random.default_rng(args.seed)
        grids = [
            _sample_motor_interior(midx, args.motor_points, rng)
            for _, midx, *_ in _MOTOR_PANELS
        ]
        for (_, midx, stem, *_), pts in zip(_MOTOR_PANELS, grids):
            csv_path = args.out_dir / f"{stem}.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["torque_Nm", "speed_rad_s"])
                writer.writerows(pts.tolist())
            print(f"[interior fill] {csv_path}  N={pts.shape[0]}")
        _save_motor_grid_figures(
            args.out_dir, grids, dpi=args.dpi, title_tag="interior fill",
        )
        return

    if args.filter_boundary:
        grids = [
            _sample_motor_envelope_dense(midx, n=args.boundary_samples)
            for _, midx, *_ in _MOTOR_PANELS
        ]
        for (_, midx, stem, *_), pts in zip(_MOTOR_PANELS, grids):
            csv_path = args.out_dir / f"{stem}.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["torque_Nm", "speed_rad_s"])
                writer.writerows(pts.tolist())
            print(f"[dense boundary] {csv_path}  N={pts.shape[0]}")
        _save_motor_grid_figures(
            args.out_dir, grids, dpi=args.dpi, title_tag="envelope boundary",
        )
        return

    rng = np.random.default_rng(args.seed)
    motor_grids: list[np.ndarray] | None = None
    if args.joint_only:
        motor_grids = _load_motor_grids_from_csv(args.out_dir)
        tau_m, omega_m = _motor_states_aligned(motor_grids)
        if args.n_points < tau_m.shape[0]:
            tau_m = tau_m[: args.n_points]
            omega_m = omega_m[: args.n_points]
        tau_j, omega_j = motor_to_joint(tau_m, omega_m)
        n = tau_m.shape[0]
        csv_path = args.out_dir / "virtual_joint_tw.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "knee_torque_Nm", "knee_speed_rad_s",
                "ankle_pitch_torque_Nm", "ankle_pitch_speed_rad_s",
                "ankle_roll_torque_Nm", "ankle_roll_speed_rad_s",
                "toe_torque_Nm", "toe_speed_rad_s",
            ])
            for i in range(n):
                writer.writerow([
                    tau_j[i, KN], omega_j[i, KN],
                    tau_j[i, AK], omega_j[i, AK],
                    tau_j[i, FT], omega_j[i, FT],
                    tau_j[i, TO], omega_j[i, TO],
                ])
        show_envelope = not args.no_envelope
        for label, col, stem in _JOINT_PANELS:
            tau = tau_j[:, col]
            omega = omega_j[:, col]
            xlim, ylim = _joint_axis_limits(tau, omega, col, show_envelope=show_envelope)
            fig, ax = plt.subplots(figsize=(5.2, 4.6))
            _panel_joint_cloud(
                ax, tau, omega, col, f"{label} — motor→joint (N={n})",
                xlim, ylim, show_envelope=show_envelope,
            )
            fig.tight_layout()
            out = args.out_dir / f"{stem}.png"
            fig.savefig(out, dpi=args.dpi)
            plt.close(fig)
            if show_envelope:
                ex, ey = _cached_joint_envelope(col)
                print(
                    f"[saved] {out}  scatter τ∈[{tau.min():.1f},{tau.max():.1f}] "
                    f"ω∈[{omega.min():.1f},{omega.max():.1f}]  "
                    f"envelope τ∈[{ex.min():.1f},{ex.max():.1f}] ω∈[{ey.min():.1f},{ey.max():.1f}]"
                )
            else:
                print(
                    f"[saved] {out}  τ∈[{tau.min():.1f},{tau.max():.1f}] "
                    f"ω∈[{omega.min():.1f},{omega.max():.1f}]"
                )
        print(f"[saved] {csv_path}  ({n} rows, aligned motor CSV rows)")
        return
    elif args.grid:
        use_grid = True
        motor_grids = [
            (_sample_motor_grid_boundary if args.boundary_only else _sample_motor_grid)(
                midx, args.grid_side, args.grid_side,
            )
            for _, midx in _LOWER_LEG_MOTORS
        ]
        _save_motor_grid_figures(args.out_dir, motor_grids, dpi=args.dpi)
        for (_, midx, stem, *_), pts in zip(_MOTOR_PANELS, motor_grids):
            motor_csv = args.out_dir / f"{stem}.csv"
            with motor_csv.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["torque_Nm", "speed_rad_s"])
                writer.writerows(pts.tolist())
            print(f"[saved] {motor_csv}  ({pts.shape[0]} rows)")
    else:
        use_grid = False

    tau_m, omega_m = sample_motor_states(
        args.n_points, rng, grid=use_grid, grid_side=args.grid_side, per_motor_grids=motor_grids,
    )
    tau_j, omega_j = motor_to_joint(tau_m, omega_m)
    n = tau_m.shape[0]

    csv_path = args.out_dir / "virtual_joint_tw.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "knee_torque_Nm", "knee_speed_rad_s",
            "ankle_pitch_torque_Nm", "ankle_pitch_speed_rad_s",
            "ankle_roll_torque_Nm", "ankle_roll_speed_rad_s",
            "toe_torque_Nm", "toe_speed_rad_s",
        ])
        for i in range(n):
            writer.writerow([
                tau_j[i, KN], omega_j[i, KN],
                tau_j[i, AK], omega_j[i, AK],
                tau_j[i, FT], omega_j[i, FT],
                tau_j[i, TO], omega_j[i, TO],
            ])

    show_envelope = not args.no_envelope
    for label, col, stem in _JOINT_PANELS:
        tau = tau_j[:, col]
        omega = omega_j[:, col]
        xlim, ylim = _joint_axis_limits(tau, omega, col, show_envelope=show_envelope)
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        _panel_joint_cloud(
            ax, tau, omega, col, f"{label} — virtual motor samples (N={n})",
            xlim, ylim, show_envelope=show_envelope,
        )
        fig.tight_layout()
        out = args.out_dir / f"{stem}.png"
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)
        if not args.no_envelope:
            ex, ey = _cached_joint_envelope(col)
            print(
                f"[saved] {out}  scatter τ∈[{tau.min():.1f},{tau.max():.1f}] "
                f"ω∈[{omega.min():.1f},{omega.max():.1f}]  "
                f"envelope τ∈[{ex.min():.1f},{ex.max():.1f}] ω∈[{ey.min():.1f},{ey.max():.1f}]"
            )
        else:
            print(
                f"[saved] {out}  τ∈[{tau.min():.1f},{tau.max():.1f}] "
                f"ω∈[{omega.min():.1f},{omega.max():.1f}]"
            )

    print(f"[saved] {csv_path}  ({n} rows)")


if __name__ == "__main__":
    main()
