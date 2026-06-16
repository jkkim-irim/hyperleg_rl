# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare right-foot GRF averaged over stance cycles (0–100 % support phase).

Pools all ``trial_*.csv`` files in each run directory, segments cycles by right
foot-strike (rising edge of ``r_grf``), phase-normalizes each cycle, and plots
the mean curve for two configurations on one axis (biomechanics-style).

Run:
    python scripts/plot_grf_stance_compare.py \\
        --equipped logs/ICCAS/toe_equipped_2026-06-09_20-19-17 \\
        --ablation logs/ICCAS/toe_ablation_2026-06-09_20-39-55 \\
        --out ICCAS/figures/fig_grf_stance_compare.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_PHASE = 101


def _read(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        r = csv.reader(f)
        hdr = [h.strip() for h in next(r)]
        data = [[float(v) for v in row] for row in r]
    arr = np.array(data, dtype=float)
    return {name: arr[:, i] for i, name in enumerate(hdr)}


def _r_grf_col(d: dict[str, np.ndarray]) -> str:
    return "r_grf" if "r_grf" in d else "r_heel_grf"


def _strikes(grf: np.ndarray, t: np.ndarray, thr: float, min_cycle_s: float) -> list[int]:
    out: list[int] = []
    last = -1e9
    for i in range(1, len(grf)):
        if grf[i - 1] <= thr < grf[i] and (t[i] - last) >= min_cycle_s:
            out.append(i)
            last = t[i]
    return out


def _support_windows(grf: np.ndarray, strikes: list[int], thr: float) -> list[tuple[int, int]]:
    """Strike index → first sample below ``thr`` (toe-off) per stance phase."""
    out: list[tuple[int, int]] = []
    for k0 in strikes:
        k1 = k0
        for i in range(k0 + 1, len(grf)):
            if grf[i] < thr:
                k1 = i
                break
        if k1 > k0:
            out.append((k0, k1))
    return out


def _collect_r_grf(csv_dir: Path, thr: float, min_cycle_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (phase_pct, cycle_stack, kept_durations) for right-foot support phase."""
    grid = np.linspace(0.0, 1.0, N_PHASE)
    stacks: list[np.ndarray] = []
    durations: list[float] = []

    sample = _read(next(csv_dir.glob("trial_*.csv")))
    col = _r_grf_col(sample)

    for path in sorted(csv_dir.glob("trial_*.csv")):
        d = _read(path)
        t = d["t_ep_s"]
        s = _strikes(d[col], t, thr, min_cycle_s)
        for k0, k1 in _support_windows(d[col], s, thr):
            dur = float(t[k1] - t[k0])
            if dur <= 0.0:
                continue
            phase = (t[k0:k1 + 1] - t[k0]) / dur
            stacks.append(np.interp(grid, phase, d[col][k0:k1 + 1]))
            durations.append(dur)

    arr = np.array(stacks)
    dur = np.array(durations)
    if arr.shape[0] < 3:
        raise SystemExit(f"{csv_dir}: only {arr.shape[0]} cycles — check --grf_thr.")

    med = float(np.median(dur))
    keep = (dur > 0.6 * med) & (dur < 1.4 * med)
    kept = arr[keep]
    kept_dur = dur[keep]
    print(
        f"[info] {csv_dir.name}: {arr.shape[0]} cycles, {kept.shape[0]} kept, "
        f"T = {kept_dur.mean():.3f} ± {kept_dur.std():.3f} s"
    )
    return grid * 100.0, kept, kept_dur


def _style_axes(ax) -> None:
    ax.set_facecolor("#eef0f2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Percent support (%)", fontsize=12, color="#1a3a52")
    ax.set_ylabel("GRF (N)", fontsize=12, color="#1a3a52")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--equipped", type=Path, required=True)
    ap.add_argument("--ablation", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("ICCAS/figures/fig_grf_stance_compare.png"))
    ap.add_argument("--grf_thr", type=float, default=5.0)
    ap.add_argument("--min_cycle_s", type=float, default=0.25)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    runs = [
        ("Toe-equipped", "#c0392b", args.equipped),
        ("Toe-ablation", "#1a1a1a", args.ablation),
    ]

    curves: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for label, _color, d in runs:
        x, stack, _ = _collect_r_grf(d, args.grf_thr, args.min_cycle_s)
        curves.append((label, _color, x, stack.mean(axis=0)))

    y_hi = max(c[3].max() for c in curves)
    y_lo = 0.0
    y_pad = 0.08 * (y_hi - y_lo)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _style_axes(ax)
    ax.set_ylim(y_lo, y_hi + y_pad)

    for label, color, x, mu in curves:
        ax.plot(x, mu, color=color, lw=2.4, label=label)

    ax.set_title(
        "Right-foot ground reaction force\naveraged over support phase",
        fontsize=13,
        color="#1a3a52",
        pad=12,
    )
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=11)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
