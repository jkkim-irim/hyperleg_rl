# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Redraw the paper's Fig. 4 CA torque-speed envelopes with measured walk data.

Reconstructs torque-speed envelopes from the HyperLeg actuator model and overlays
operating points logged by ``play.py --fig4_operating_point``:

    fig04_motor_KN … fig04_motor_TO  ← motor_tw.csv (KN / AK / FT / TO)
    fig04a/b                        ← aliases of AK / FT (paper naming)
    fig04c_joint_AnklePitch         ← joint_tw pitch + Minkowski J envelope
    fig04d_joint_AnkleRoll          ← joint_tw roll + Minkowski J envelope
    fig04c/d_*_step4.png            ← step-4 Minkowski envelope only (diagnostic)

Run:
    python scripts/plot_fig04_ca_envelopes.py \\
        --csv_dir logs/ICCAS/Fig4/2026-06-13_16-31-01
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Per-leg Jacobian J = ∂motor/∂joint, canonical [HY HR HP KN AK FT TO] (hyperleg.py).
J = np.array([
    [25.0,  0.0,  0.0,   0.0,    0.0,    0.0,    0.0],
    [0.0,  25.0,  0.0,   0.0,    0.0,    0.0,    0.0],
    [0.0,   0.0, 25.0,   0.0,    0.0,    0.0,    0.0],
    [0.0,   0.0,  0.0, -31.11,   0.0,    0.0,    0.0],
    [0.0,   0.0,  0.0, -31.11,  26.69,  16.51,   0.0],
    [0.0,   0.0,  0.0, -31.11,  26.69, -16.51,   0.0],
    [0.0,   0.0,  0.0, -31.11,  19.23,   0.0,   21.27],
])
_ROLES = ("HY", "HR", "HP", "KN", "AK", "FT", "TO")
KN, AK, FT, TO = 3, 4, 5, 6  # canonical motor indices

# Per-motor colours on step-3 joint projection panels.
_PITCH_PROJ_STYLES: tuple[tuple[str, int, str], ...] = (
    ("KN", KN, "#1565c0"),
    ("AK", AK, "#2e7d32"),
    ("FT", FT, "#e65100"),
    ("TO", TO, "#6a1b9a"),
)
_ROLL_PROJ_STYLES: tuple[tuple[str, int, str], ...] = (
    ("AK", AK, "#2e7d32"),
    ("FT", FT, "#e65100"),
)

# Per-motor DC limits [τ_stall=τ_cont Nm, ω₀ rad/s], canonical [HY..TO].
_MOTOR_SPEC: tuple[tuple[float, float], ...] = (
    (5.04, 300.0),
    (5.04, 300.0),
    (5.04, 300.0),
    (2.68, 327.2),
    (1.63, 366.0),
    (1.63, 366.0),
    (1.63, 366.0),
)

# Palette mirroring the originals.
GREEN_LINE, GREEN_FILL = "#2e7d32", "#79c879"
PURPLE_LINE, PURPLE_FILL = "#5e35b1", "#9d8bd0"
RED_Q, BLUE_Q = "#f6d4d4", "#d4ddf6"   # motoring (τω≥0) / regenerative (τω<0)
SCATTER = "#e8590c"                    # measured operating points
ENV_LS = (0, (2.5, 1.5))               # dense dashed envelope (reads solid at print DPI)

_HIP_MOTOR_XLIM = (-5.1, 5.1)
_HIP_MOTOR_YLIM = (-650, 650)
_HIP_MOTOR_YTICKS = [-600, -400, -200, 0, 200, 400, 600]
_ANKLE_MOTOR_XLIM = (-1.95, 1.95)
_ANKLE_MOTOR_YLIM = (-800, 800)
_ANKLE_MOTOR_YTICKS = [-800, -600, -400, -200, 0, 200, 400, 600, 800]
_KNEE_MOTOR_XLIM = (-2.95, 2.95)
_KNEE_MOTOR_YLIM = (-700, 700)
_KNEE_MOTOR_YTICKS = [-700, -500, -300, -100, 0, 100, 300, 500, 700]

# (fig stem, τ CSV col, ω CSV col, canonical motor idx, legend label, xlim, ylim, xticks, yticks)
_MOTOR_PANELS: tuple[tuple, ...] = (
    ("fig04_motor_KN", "knee_torque_Nm", "knee_speed_rad_s", KN,
     "Knee motor (KN)", _KNEE_MOTOR_XLIM, _KNEE_MOTOR_YLIM, [-2, 0, 2], _KNEE_MOTOR_YTICKS),
    ("fig04_motor_AK", "L_ankle_torque_Nm", "L_ankle_speed_rad_s", AK,
     "Ankle Pitch motor (AK)", _ANKLE_MOTOR_XLIM, _ANKLE_MOTOR_YLIM, [-1, 0, 1], _ANKLE_MOTOR_YTICKS),
    ("fig04_motor_FT", "R_ankle_torque_Nm", "R_ankle_speed_rad_s", FT,
     "Ankle Roll motor (FT)", _ANKLE_MOTOR_XLIM, _ANKLE_MOTOR_YLIM, [-1, 0, 1], _ANKLE_MOTOR_YTICKS),
    ("fig04_motor_TO", "toe_torque_Nm", "toe_speed_rad_s", TO,
     "Toe motor (TO)", _ANKLE_MOTOR_XLIM, _ANKLE_MOTOR_YLIM, [-1, 0, 1], _ANKLE_MOTOR_YTICKS),
)
# Paper Fig. 4 aliases (same data as AK / FT panels above).
_MOTOR_PANEL_ALIASES: tuple[tuple[str, str], ...] = (
    ("fig04a_L_ankle_motor", "fig04_motor_AK"),
    ("fig04b_R_ankle_motor", "fig04_motor_FT"),
)


def _motor_limits(motor_idx: int) -> tuple[float, float, float]:
    t_s, w0 = _MOTOR_SPEC[motor_idx]
    wc = w0 * (1.0 + t_s / t_s)
    return t_s, w0, wc


def _motor_max_tau(omega: np.ndarray, motor_idx: int) -> np.ndarray:
    t_s, w0, wc = _motor_limits(motor_idx)
    om = np.clip(omega, -wc, wc)
    return np.minimum(t_s * (1.0 - om / w0), t_s)


def _motor_min_tau(omega: np.ndarray, motor_idx: int) -> np.ndarray:
    t_s, w0, wc = _motor_limits(motor_idx)
    om = np.clip(omega, -wc, wc)
    return np.maximum(t_s * (-1.0 - om / w0), -t_s)


def motor_envelope(motor_idx: int = AK, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Single-motor rhombus boundary as a closed (torque, speed) polygon."""
    _, _, wc = _motor_limits(motor_idx)
    w = np.linspace(-wc, wc, n)
    up, lo = _motor_max_tau(w, motor_idx), _motor_min_tau(w, motor_idx)
    xs = np.concatenate([up, lo[::-1]])
    ys = np.concatenate([w, w[::-1]])
    return xs, ys


def joint_envelope(col: int, n_motor: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Joint (τ, ω) envelope = Minkowski sum of DC motor polygons projected through J."""
    j_inv_t = np.linalg.inv(J).T
    motors = sorted(
        m for m in range(J.shape[0])
        if J[m, col] != 0.0 or j_inv_t[m, col] != 0.0
    )
    polys = []
    for m in motors:
        tx, ty = motor_envelope(m, n=n_motor)
        polys.append(np.column_stack([J[m, col] * tx, j_inv_t[m, col] * ty]))
    hull = _convex_hull_ordered(_minkowski_sum_vertices(polys))
    return hull[:, 0], hull[:, 1]


def _motor_rhombus_vertices(motor_idx: int) -> np.ndarray:
    """Rhombus corners in motor (τ_m, ω_m): (±τ_stall, 0) and (0, ±ω_c)."""
    t_s, _, wc = _motor_limits(motor_idx)
    return np.array([[t_s, 0.0], [-t_s, 0.0], [0.0, wc], [0.0, -wc]], dtype=float)


def motor_envelope_on_joint(motor_idx: int, col: int) -> tuple[np.ndarray, np.ndarray]:
    """Step 3: map one motor rhombus (τ_m, ω_m) → joint (γ·τ_m, δ·ω_m); no sum across motors."""
    j_inv_t = np.linalg.inv(J).T
    gamma_tau = J[motor_idx, col]
    gamma_omega = j_inv_t[motor_idx, col]
    tau_m, omega_m = motor_envelope(motor_idx)
    return gamma_tau * tau_m, gamma_omega * omega_m


def joint_per_motor_envelopes(
    col: int,
    styles: tuple[tuple[str, int, str], ...],
) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    """One mapped rhombus per contributing motor (step 3, no Minkowski sum)."""
    out: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    j_inv_t = np.linalg.inv(J).T
    for role, midx, color in styles:
        if J[midx, col] == 0.0 and j_inv_t[midx, col] == 0.0:
            continue
        tx, ty = motor_envelope_on_joint(midx, col)
        out.append((role, tx, ty, color))
    return out


def _motor_joint_vertices(motor_idx: int, col: int, j_inv_t: np.ndarray) -> np.ndarray:
    """Motor rhombus mapped to joint (τ_j, ω_j) via τ_j = J[m,col]·τ_m, ω_j = (J^{-T})[m,col]·ω_m."""
    gamma_tau = J[motor_idx, col]
    gamma_omega = j_inv_t[motor_idx, col]
    verts = _motor_rhombus_vertices(motor_idx)
    return np.column_stack([gamma_tau * verts[:, 0], gamma_omega * verts[:, 1]])


def _minkowski_sum_vertices(polys: list[np.ndarray]) -> np.ndarray:
    """Vertex sums of convex polygons; hull after each pair keeps cost bounded."""
    acc = polys[0]
    for poly in polys[1:]:
        acc = acc[:, None, :] + poly[None, :, :]
        acc = acc.reshape(-1, 2)
        acc = _convex_hull_ordered(acc)
    return acc


def _convex_hull_ordered(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise ordered vertices of the 2-D convex hull (monotone chain)."""
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 0], pts[:, 1]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1])
    return hull


def _read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a CSV into (header, (N, C) float array)."""
    with path.open() as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = np.array([[float(v) for v in r] for r in rows[1:]], dtype=float)
    return header, data


def _panel(
    ax,
    env,
    line_c,
    fill_c,
    tau,
    spd,
    label,
    xlim,
    ylim,
    xticks,
    yticks,
):
    """Draw one envelope + quadrant shading + scatter on ``ax``."""
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

    ex, ey = env
    ax.fill(ex, ey, facecolor=fill_c, alpha=0.35, zorder=1)
    ax.plot(np.append(ex, ex[0]), np.append(ey, ey[0]), color=line_c, lw=1.0,
            ls=ENV_LS, zorder=3, label=label)

    ax.scatter(tau, spd, s=8, c=SCATTER, alpha=0.7, edgecolors="none", zorder=2,
               label="Operating points")

    tag = dict(color="0.45", fontsize=12, fontweight="bold", zorder=4)
    pad_x, pad_y = 0.04 * (xmax - xmin), 0.05 * (ymax - ymin)
    ax.text(xmin + pad_x, ymax - pad_y, "Q2", ha="left", va="top", **tag)
    ax.text(xmin + pad_x, ymin + pad_y, "Q3", ha="left", va="bottom", **tag)
    ax.text(xmax - pad_x, ymin + pad_y, "Q4", ha="right", va="bottom", **tag)

    ax.set_xlabel("Torque [Nm]")
    ax.set_ylabel("Speed [rad/s]")
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [handles[labels.index(label)], handles[labels.index("Operating points")]],
        [label, "Operating points"],
        loc="upper right", framealpha=0.9, fontsize=9,
    )


def _format_gamma_tau(v: float) -> str:
    """Format J[m,joint] — literals from hyperleg.py ``_J_PER_LEG``."""
    if v == 0.0:
        return "0"
    return f"{v:.2f}"


def _format_gamma_omega(v: float) -> str:
    """Format (J^{-T})[m,joint] — derived from ``_J_PER_LEG``, not a matrix entry."""
    if v == 0.0:
        return "0"
    return f"{v:.12g}"


def _proj_legend(role: str, motor_idx: int, col: int, j_inv_t: np.ndarray) -> str:
    g_t = J[motor_idx, col]
    g_w = j_inv_t[motor_idx, col]
    return f"{role}  (γτ={_format_gamma_tau(g_t)}, γω={_format_gamma_omega(g_w)})"


def _panel_joint_step3(
    ax,
    col: int,
    title: str,
    styles: tuple[tuple[str, int, str], ...],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xticks: list[float],
    yticks: list[float],
) -> None:
    """Joint plane: each motor rhombus mapped via J (step 3, no Minkowski sum)."""
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

    j_inv_t = np.linalg.inv(J).T
    for role, tx, ty, color in joint_per_motor_envelopes(col, styles):
        midx = _ROLES.index(role)
        ax.fill(tx, ty, facecolor=color, alpha=0.2, zorder=1)
        ax.plot(
            np.append(tx, tx[0]),
            np.append(ty, ty[0]),
            color=color,
            lw=1.2,
            ls=ENV_LS,
            zorder=3,
            label=_proj_legend(role, midx, col, j_inv_t),
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


def _panel_joint_step4(
    ax,
    col: int,
    title: str,
    legend_label: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xticks: list[float],
    yticks: list[float],
) -> None:
    """Joint envelope = Minkowski sum of per-motor rhombi projected through J (step 4)."""
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

    ex, ey = joint_envelope(col)
    ax.fill(ex, ey, facecolor=PURPLE_FILL, alpha=0.35, zorder=1)
    ax.plot(
        np.append(ex, ex[0]),
        np.append(ey, ey[0]),
        color=PURPLE_LINE,
        lw=1.2,
        ls=ENV_LS,
        zorder=3,
        label=legend_label,
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
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--csv_dir", type=Path,
                        default=repo / "logs/ICCAS/toe_equipped_2026-05-30_18-23-30",
                        help="Directory holding motor_tw.csv and joint_tw.csv.")
    parser.add_argument("--out_dir", type=Path, default=repo / "ICCAS/figures",
                        help="Where to write fig04*.png files.")
    parser.add_argument("--dpi", type=int, default=400, help="Output PNG resolution.")
    args = parser.parse_args()

    motor_header, motor = _read_csv(args.csv_dir / "motor_tw.csv")
    _, joint = _read_csv(args.csv_dir / "joint_tw.csv")

    def mcols(tau_name: str, omega_name: str) -> tuple[np.ndarray, np.ndarray]:
        ti, wi = motor_header.index(tau_name), motor_header.index(omega_name)
        return motor[:, ti], motor[:, wi]

    pitch_env = joint_envelope(AK)
    roll_env = joint_envelope(FT)
    px, py = pitch_env
    rx, ry = roll_env
    print(
        f"[pitch envelope] τ∈[{px.min():.1f},{px.max():.1f}] "
        f"ω∈[{py.min():.1f},{py.max():.1f}]"
    )
    print(
        f"[roll  envelope] τ∈[{rx.min():.1f},{rx.max():.1f}] "
        f"ω∈[{ry.min():.1f},{ry.max():.1f}]"
    )

    plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.0})

    panels: list[tuple] = []
    saved_motor_figs: dict[str, Path] = {}

    for stem, tau_col, omega_col, midx, label, xlim, ylim, xt, yt in _MOTOR_PANELS:
        if tau_col not in motor_header:
            print(f"[skip] {stem}: {tau_col} not in motor_tw.csv")
            continue
        tau, spd = mcols(tau_col, omega_col)
        panels.append((stem, motor_envelope(midx), GREEN_LINE, GREEN_FILL,
                       tau, spd, label, xlim, ylim, xt, yt))

    panels.extend([
        ("fig04c_joint_AnklePitch", pitch_env, PURPLE_LINE, PURPLE_FILL,
         joint[:, 0], joint[:, 1],
         "Ankle Pitch joint", (-138, 138), (-60, 60),
         [-100, -50, 0, 50, 100], [-60, -40, -20, 0, 20, 40, 60]),
        ("fig04d_joint_AnkleRoll", roll_env, PURPLE_LINE, PURPLE_FILL,
         joint[:, 2], joint[:, 3],
         "Ankle Roll joint", (-62, 62), (-60, 60),
         [-60, -40, -20, 0, 20, 40, 60], [-60, -40, -20, 0, 20, 40, 60]),
    ])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, env, lc, fc, tau, spd, label, xlim, ylim, xt, yt in panels:
        fig, ax = plt.subplots(figsize=(5.0, 4.4))
        _panel(ax, env, lc, fc, tau, spd, label, xlim, ylim, xt, yt)
        fig.tight_layout()
        out = args.out_dir / f"{name}.png"
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)
        if name.startswith("fig04_motor_"):
            saved_motor_figs[name] = out
        print(f"[saved] {out}  (scatter N={len(tau)}, dpi={args.dpi})")

    for alias, source in _MOTOR_PANEL_ALIASES:
        src = saved_motor_figs.get(source)
        if src is None:
            continue
        dst = args.out_dir / f"{alias}.png"
        dst.write_bytes(src.read_bytes())
        print(f"[saved] {dst}  (alias of {source})")

    pitch_xlim = (-138, 138)
    pitch_ylim = (-60, 60)
    pitch_xt = [-100, -50, 0, 50, 100]
    pitch_yt = [-60, -40, -20, 0, 20, 40, 60]
    roll_xlim = (-62, 62)
    roll_ylim = (-60, 60)
    roll_xt = [-60, -40, -20, 0, 20, 40, 60]

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    _panel_joint_step3(
        ax, AK, "Ankle Pitch — per-motor J projection (step 3)",
        _PITCH_PROJ_STYLES, pitch_xlim, pitch_ylim, pitch_xt, pitch_yt,
    )
    fig.tight_layout()
    step3_out = args.out_dir / "fig04c_joint_AnklePitch_step3.png"
    fig.savefig(step3_out, dpi=args.dpi)
    plt.close(fig)
    print(f"[saved] {step3_out}  (per-motor mapped envelopes only, dpi={args.dpi})")

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    _panel_joint_step3(
        ax, FT, "Ankle Roll — per-motor J projection (step 3)",
        _ROLL_PROJ_STYLES, roll_xlim, roll_ylim, roll_xt, roll_xt,
    )
    fig.tight_layout()
    roll_step3_out = args.out_dir / "fig04d_joint_AnkleRoll_step3.png"
    fig.savefig(roll_step3_out, dpi=args.dpi)
    plt.close(fig)
    print(f"[saved] {roll_step3_out}  (per-motor mapped envelopes only, dpi={args.dpi})")

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    _panel_joint_step4(
        ax, AK,
        "Ankle Pitch — Minkowski sum (step 4)",
        "Ankle Pitch joint (KN⊕AK⊕FT⊕TO)",
        pitch_xlim, pitch_ylim, pitch_xt, pitch_yt,
    )
    fig.tight_layout()
    step4_out = args.out_dir / "fig04c_joint_AnklePitch_step4.png"
    fig.savefig(step4_out, dpi=args.dpi)
    plt.close(fig)
    tx, ty = joint_envelope(AK)
    print(
        f"[saved] {step4_out}  (Minkowski envelope only, "
        f"τ∈[{tx.min():.1f},{tx.max():.1f}] ω∈[{ty.min():.1f},{ty.max():.1f}], dpi={args.dpi})"
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    _panel_joint_step4(
        ax, FT,
        "Ankle Roll — Minkowski sum (step 4)",
        "Ankle Roll joint (AK⊕FT)",
        roll_xlim, roll_ylim, roll_xt, roll_xt,
    )
    fig.tight_layout()
    roll_step4_out = args.out_dir / "fig04d_joint_AnkleRoll_step4.png"
    fig.savefig(roll_step4_out, dpi=args.dpi)
    plt.close(fig)
    rx, ry = joint_envelope(FT)
    print(
        f"[saved] {roll_step4_out}  (Minkowski envelope only, "
        f"τ∈[{rx.min():.1f},{rx.max():.1f}] ω∈[{ry.min():.1f},{ry.max():.1f}], dpi={args.dpi})"
    )


if __name__ == "__main__":
    main()
