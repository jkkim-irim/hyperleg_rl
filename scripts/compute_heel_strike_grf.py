# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Heel-strike GRF from ``play_goto_x --log_csv`` trial CSVs.

Each rising edge of ``l_grf`` / ``r_grf`` (heel-only, > ``--grf_thr``) is one
heel strike. Per strike we report:

  * **first** — GRF at the first contact sample (rising-edge row)
  * **peak50** — max GRF in [strike, strike + 50 ms]
  * **peak100** — max GRF in [strike, strike + 100 ms]

Trial rows are means over all strikes; ``total_avg`` pools every strike in the run.

Expect ``l_grf`` / ``r_grf`` to be ``||F_heel||`` (see ``PowerCSVLogger``).

Run:
    python scripts/compute_heel_strike_grf.py --csv_dir logs/ICCAS/<run>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

_METRICS = ("first", "peak50", "peak100")
_PEAK_WINDOWS_MS = (50.0, 100.0)


def _read(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        r = csv.reader(f)
        hdr = [h.strip() for h in next(r)]
        data = [[float(v) for v in row] for row in r]
    arr = np.array(data, dtype=float)
    return {name: arr[:, i] for i, name in enumerate(hdr)}


def _strike_metrics(
    grf: np.ndarray, t: np.ndarray, thr: float, min_cycle_s: float
) -> dict[str, np.ndarray]:
    """Per-strike first-sample and post-strike peak GRF."""
    buckets: dict[str, list[float]] = {m: [] for m in _METRICS}
    last = -1e9
    for i in range(1, len(grf)):
        if grf[i - 1] <= thr < grf[i] and (t[i] - last) >= min_cycle_s:
            buckets["first"].append(float(grf[i]))
            for w_ms, key in zip(_PEAK_WINDOWS_MS, ("peak50", "peak100")):
                win = (t >= t[i]) & (t <= t[i] + w_ms * 1e-3)
                buckets[key].append(float(grf[win].max()))
            last = t[i]
    return {k: np.array(v, dtype=float) for k, v in buckets.items()}


def _trial_means(d: dict[str, np.ndarray]) -> dict[str, float]:
    return {m: float(d[m].mean()) for m in _METRICS}


def _summarize(run: Path, thr: float, min_cycle_s: float) -> None:
    trial_rows: list[dict] = []
    pooled: dict[str, dict[str, list[float]]] = {
        "l": {m: [] for m in _METRICS},
        "r": {m: [] for m in _METRICS},
    }

    for path in sorted(run.glob("trial_*.csv"), key=lambda p: int(p.stem.split("_")[1])):
        d = _read(path)
        t = d["t_ep_s"]
        lm = _strike_metrics(d["l_grf"], t, thr, min_cycle_s)
        rm = _strike_metrics(d["r_grf"], t, thr, min_cycle_s)
        n = len(lm["first"])
        trial_rows.append({
            "trial": int(path.stem.split("_")[1]),
            "l": _trial_means(lm),
            "r": _trial_means(rm),
            "l_strikes": n,
            "r_strikes": len(rm["first"]),
        })
        for foot, mdict in (("l", lm), ("r", rm)):
            for m in _METRICS:
                pooled[foot][m].extend(mdict[m].tolist())

    print(f"\n=== {run.name} (heel-strike GRF, thr={thr} N) ===")
    header = (
        "trial\tl_first\tr_first\tl_peak50\tr_peak50\tl_peak100\tr_peak100"
        "\tl_strikes\tr_strikes"
    )
    print(header)
    for row in trial_rows:
        l, r = row["l"], row["r"]
        print(
            f"{row['trial']}\t{l['first']}\t{r['first']}\t{l['peak50']}\t{r['peak50']}"
            f"\t{l['peak100']}\t{r['peak100']}\t{row['l_strikes']}\t{row['r_strikes']}"
        )
    l_tot = {m: np.mean(pooled["l"][m]) for m in _METRICS}
    r_tot = {m: np.mean(pooled["r"][m]) for m in _METRICS}
    print(
        f"total_avg\t{l_tot['first']}\t{r_tot['first']}\t{l_tot['peak50']}\t{r_tot['peak50']}"
        f"\t{l_tot['peak100']}\t{r_tot['peak100']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv_dir", type=Path, nargs="+", required=True)
    ap.add_argument("--grf_thr", type=float, default=5.0)
    ap.add_argument("--min_cycle_s", type=float, default=0.25)
    args = ap.parse_args()
    for d in args.csv_dir:
        _summarize(d, args.grf_thr, args.min_cycle_s)


if __name__ == "__main__":
    main()
