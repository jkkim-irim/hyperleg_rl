# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Gait-cycle phase-averaged plots from play --log_csv trial CSVs.

A single time window is one realization; averaging many gait cycles is far more
trustworthy. This script:

  1. Pools every ``trial_*.csv`` in a run directory.
  2. Detects right foot-strikes (rising edge of ``r_grf``) → one cycle =
     strike[k] .. strike[k+1].
  3. Phase-normalizes each cycle to 0-100 % (linear resample to 101 points), so
     cycles of unequal duration are comparable.
  4. Averages across all kept cycles (outlier-duration cycles dropped) and plots
     mean ± 1 SD bands vs % gait cycle (0 % = right foot-strike):

       <run>/cycle_avg_joint_torque.png   hip/knee/ankle/toe (right leg, signed)
       <run>/cycle_avg_joint_power.png    hip/knee/ankle/toe battery power
       <run>/cycle_avg_power_breakdown.png joule / mech / total (filled)
       <run>/cycle_avg_grf.png            l_grf / r_grf

Run:
    python scripts/plot_gait_cycle_average.py --csv_dir logs/ICCAS/<run>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUP_STYLE = (
    ("hip", "Hip", "#2440c8"),
    ("knee", "Knee", "#e01b1b"),
    ("ankle", "Ankle", "#1c9e3b"),
    ("toe", "Toe", "#c92ec9"),
)
N_PHASE = 101  # resample points per cycle (0..100 %)


def _read(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        r = csv.reader(f)
        hdr = [h.strip() for h in next(r)]
        data = [[float(v) for v in row] for row in r]
    arr = np.array(data, dtype=float)
    return {name: arr[:, i] for i, name in enumerate(hdr)}


def _grf_cols(d: dict[str, np.ndarray]) -> tuple[str, str]:
    """Resolve GRF column names (new ``l_grf``/``r_grf`` or legacy heel-only)."""
    if "l_grf" in d:
        return "l_grf", "r_grf"
    return "l_heel_grf", "r_heel_grf"


def _strikes(grf: np.ndarray, t: np.ndarray, thr: float, min_cycle_s: float) -> list[int]:
    """Indices where foot GRF rises through ``thr`` (debounced by min_cycle_s)."""
    out: list[int] = []
    last = -1e9
    for i in range(1, len(grf)):
        if grf[i - 1] <= thr < grf[i] and (t[i] - last) >= min_cycle_s:
            out.append(i)
            last = t[i]
    return out


def _collect(csv_dir: Path, signals: list[str], r_grf_col: str, thr: float, min_cycle_s: float):
    """Phase-normalize every cycle in every trial CSV; return stacks + durations."""
    grid = np.linspace(0.0, 1.0, N_PHASE)
    stacks: dict[str, list[np.ndarray]] = {s: [] for s in signals}
    durations: list[float] = []
    for path in sorted(csv_dir.glob("trial_*.csv")):
        d = _read(path)
        t = d["t_ep_s"]
        for k0, k1 in zip(s := _strikes(d[r_grf_col], t, thr, min_cycle_s), s[1:]):
            dur = float(t[k1] - t[k0])
            if dur <= 0.0:
                continue
            phase = (t[k0:k1 + 1] - t[k0]) / dur
            durations.append(dur)
            for sig in signals:
                stacks[sig].append(np.interp(grid, phase, d[sig][k0:k1 + 1]))
    return grid, stacks, np.array(durations)


def _arrow_axes(ax, ylabel: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_position(("data", 0.0))
    ax.spines["left"].set_linewidth(1.3)
    ax.set_xlim(0.0, 104.0)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, ms=7, zorder=6)
    ax.annotate("", xy=(104.0, 0.0), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.3), annotation_clip=False, zorder=6)
    ax.tick_params(axis="x", length=3)
    ax.grid(True, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel("Gait cycle [%]", loc="right")
    ax.set_ylabel(ylabel, loc="top")


def _band(ax, x, arr, color, label):
    """Mean line + ±1 SD shaded band for a (n_cycles, N) stack."""
    mu, sd = arr.mean(axis=0), arr.std(axis=0)
    ax.fill_between(x, mu - sd, mu + sd, facecolor=color, alpha=0.15, zorder=2)
    ax.plot(x, mu, color=color, lw=2.0, label=label, zorder=3)
    return mu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv_dir", type=Path, nargs="+", required=True,
                    help="Run dir(s) with trial_*.csv. Pass two to match their y-axes.")
    ap.add_argument("--grf_thr", type=float, default=5.0, help="Foot-strike GRF threshold [N].")
    ap.add_argument("--min_cycle_s", type=float, default=0.25, help="Debounce / min cycle [s].")
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.0})

    # Collect phase-averaged cycles for every run dir.
    runs = []
    for d in args.csv_dir:
        sample = _read(next(d.glob("trial_*.csv")))
        l_grf_col, r_grf_col = _grf_cols(sample)
        has_toe = "toe_power_W" in sample
        groups = [g for g in GROUP_STYLE if g[0] != "toe" or has_toe]
        signals = (
            [f"{k}_torque_Nm" for k, *_ in groups] + [f"{k}_power_W" for k, *_ in groups]
            + ["joule_R_W", "mech_R_W", l_grf_col, r_grf_col]
        )
        grid, stacks, durations = _collect(d, signals, r_grf_col, args.grf_thr, args.min_cycle_s)
        if durations.size < 3:
            raise SystemExit(f"{d}: only {durations.size} cycles — check --grf_thr.")
        med = float(np.median(durations))
        keep = (durations > 0.6 * med) & (durations < 1.4 * med)
        arr = {s: np.array(v)[keep] for s, v in stacks.items()}
        arr["total_R"] = arr["joule_R_W"] + arr["mech_R_W"]
        kept = durations[keep]
        variant = "Toe" if has_toe else "No-toe"
        runs.append({
            "dir": d, "x": grid * 100.0, "arr": arr, "groups": groups,
            "l_grf_col": l_grf_col, "r_grf_col": r_grf_col,
            "sub": f"{variant} (right leg) — {int(keep.sum())} cycles,  "
                   f"T = {kept.mean():.3f} ± {kept.std():.3f} s",
        })
        print(f"[info] {d.name}: {durations.size} cycles, {int(keep.sum())} kept (T~{med:.3f}s).")

    def ylim(cols: list[str]) -> tuple[float, float]:
        """Shared y-limits across all runs for the given mean±std bands."""
        lo = min((r["arr"][c].mean(0) - r["arr"][c].std(0)).min()
                 for r in runs for c in cols if c in r["arr"])
        hi = max((r["arr"][c].mean(0) + r["arr"][c].std(0)).max()
                 for r in runs for c in cols if c in r["arr"])
        pad = 0.06 * (hi - lo)
        return lo - pad, hi + pad

    tq_lim = ylim([f"{k}_torque_Nm" for k, *_ in GROUP_STYLE])
    pw_lim = ylim([f"{k}_power_W" for k, *_ in GROUP_STYLE])
    bd_lim = ylim(["joule_R_W", "mech_R_W", "total_R"])
    grf_lim = ylim(["l_grf", "r_grf", "l_heel_grf", "r_heel_grf"])

    def new_ax(ylabel, lim):
        fig, ax = plt.subplots(figsize=(4.9, 4.9))
        _arrow_axes(ax, ylabel)
        ax.set_ylim(*lim)
        ax.set_box_aspect(1)  # square plot area
        return fig, ax

    for r in runs:
        x, arr, d = r["x"], r["arr"], r["dir"]

        fig, ax = new_ax("Torque [Nm]", tq_lim)
        for key, label, color in r["groups"]:
            _band(ax, x, arr[f"{key}_torque_Nm"], color, label)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
        fig.tight_layout(); fig.savefig(d / "cycle_avg_joint_torque.png", dpi=args.dpi); plt.close(fig)

        fig, ax = new_ax("Power [W]", pw_lim)
        for key, label, color in r["groups"]:
            _band(ax, x, arr[f"{key}_power_W"], color, label)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
        fig.tight_layout(); fig.savefig(d / "cycle_avg_joint_power.png", dpi=args.dpi); plt.close(fig)

        fig, ax = new_ax("Power [W]", bd_lim)
        tot_mu = arr["total_R"].mean(axis=0)
        ax.fill_between(x, tot_mu, 0.0, facecolor="#8fdc8f", alpha=0.5, zorder=1, label="Total")
        _band(ax, x, arr["joule_R_W"], "#f08a6c", "Joule heating")
        _band(ax, x, arr["mech_R_W"], "#3b6fd6", "Mechanical power")
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
        fig.tight_layout(); fig.savefig(d / "cycle_avg_power_breakdown.png", dpi=args.dpi); plt.close(fig)

        fig, ax = new_ax("Foot GRF [N]", grf_lim)
        _band(ax, x, arr[r["l_grf_col"]], "#2440c8", "L foot")
        _band(ax, x, arr[r["r_grf_col"]], "#e01b1b", "R foot")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
        fig.tight_layout(); fig.savefig(d / "cycle_avg_grf.png", dpi=args.dpi); plt.close(fig)

        print(f"[saved] 4 cycle-average PNGs under {d}")


if __name__ == "__main__":
    main()
