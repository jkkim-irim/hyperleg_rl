# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Live telemetry plotter for HyperLeg motors (Play mode only).

Three ImPlot tabs:
- "Motors": stacked joint-side torque / motor_heat subplots, each with a role
  selector (HY/HR/HP/KN/AK/FT/TO); both L and R legs are drawn together so
  bilateral asymmetry is visible.
- "Power": per-joint-group power consumption (Joule + Mech) coloured by group,
  plus a Joule heating / Mechanical power / Total(sum) breakdown. A leg
  selector picks L, R, or L+R. Mechanical power is signed (+ motoring in
  Q1/Q3, - regen in Q2/Q4); Joule heating is always >=0 across all quadrants.
- "GRF": per-foot heel ground-reaction-force magnitude — current value, a
  bar gauge, a peak-since-launch tracker per side (updates instantly when a
  new peak is hit; flashes for ~1 s after each update), and the time series.
  Reuses ``heel_grf_magnitude`` from the tasks package so the value matches
  the obs/reward definitions exactly.

Designed to not bottleneck the sim main loop:
- Per env.step() the sim thread copies one env-row of two tensors per leg to
  CPU and appends to numpy ring buffers under a short-held lock.
- A daemon thread runs the GLFW/ImPlot window via ``imgui_bundle.immapp.run``;
  it copies a snapshot under the lock then renders without holding it.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from hyperleg_rl.actuators.coupled_leg import CoupledLegActuator


# Canonical role order matches CoupledLegActuator._ROLE_SUFFIXES.
_ALL_ROLES: tuple[str, ...] = ("HY", "HR", "HP", "KN", "AK", "FT", "TO")
_SIGNALS: tuple[str, ...] = ("applied_tau", "motor_heat", "joule", "mech")

# Subplot configuration for the "Motors" tab: (key, signal, title, y-axis label).
_SUBPLOTS: tuple[tuple[str, str, str, str], ...] = (
    ("top", "applied_tau", "Joint-side torque (applied_effort)", "tau_j [Nm]"),
    ("bot", "motor_heat", "Motor heat EMA  (tau_m / tau_cont)^2", "heat [-]"),
)

# Per-motor electrical power coefficient Cp = R_LL / K_t^2 [W/Nm^2] by canonical role.
# SOURCE OF TRUTH: tasks/.../hyperleg/mdp/rewards.py::_POWER_COEF_BY_ROLE (datasheet-
# derived). Duplicated here — with the same asymmetric torque/efficiency scaling as
# _compute_cot_components — so the viz package stays independent of the task package.
# Keep in sync if rewards.py changes.
_POWER_COEF_BY_ROLE: dict[str, float] = {
    "HY": 3.54148162, "HR": 3.54148162, "HP": 3.54148162,
    "KN": 33.44203328, "AK": 124.567474, "FT": 124.567474, "TO": 124.567474,
}

# Per-group power: member roles are summed into one line each (label, roles, RGBA):
# Hip = HY+HR+HP (blue), Knee = KN (red), Ankle = AK+FT (green), Toe = TO (pink-purple).
_POWER_GROUPS: tuple[tuple[str, tuple[str, ...], tuple[float, float, float, float]], ...] = (
    ("Hip", ("HY", "HR", "HP"), (0.20, 0.55, 1.00, 1.0)),
    ("Knee", ("KN",), (1.00, 0.30, 0.25, 1.0)),
    ("Ankle", ("AK", "FT"), (0.20, 0.80, 0.35, 1.0)),
    ("Toe", ("TO",), (0.85, 0.40, 0.95, 1.0)),
)


class MotorTelemetryPlotter:
    """Live ImPlot window showing per-joint motor telemetry for env 0.

    Args:
        env: A wrapped Isaac Lab env. ``env.unwrapped.scene["robot"].actuators``
            must contain ``"left_leg"`` and ``"right_leg"`` keyed
            ``CoupledLegActuator`` instances.
        capacity: Ring-buffer length. Default 250 ≈ 5 s at 50 Hz control rate.
        env_index: Which env to plot. Defaults to 0.
    """

    def __init__(self, env, *, capacity: int = 250, env_index: int = 0) -> None:
        robot = env.unwrapped.scene["robot"]
        try:
            left_act = robot.actuators["left_leg"]
            right_act = robot.actuators["right_leg"]
        except KeyError as exc:
            raise RuntimeError(
                "MotorTelemetryPlotter expects actuators keyed 'left_leg' / 'right_leg' "
                f"on robot; found {list(robot.actuators.keys())}."
            ) from exc

        self._actuators: dict[str, "CoupledLegActuator"] = {"L": left_act, "R": right_act}
        n_left = len(left_act._joint_names)
        n_right = len(right_act._joint_names)
        if n_left != n_right or n_left not in (6, 7):
            raise ValueError(
                f"Unexpected per-leg joint count: L={n_left}, R={n_right} (expected 6 or 7, equal)."
            )
        self._n_roles = n_left
        self._roles = list(_ALL_ROLES[:n_left])
        self._env_index = int(env_index)
        # POWER_COEF in canonical role order [:n] for per-joint Joule heating.
        self._power_coef = np.array([_POWER_COEF_BY_ROLE[r] for r in self._roles], dtype=np.float32)
        # Resolve power-plot groups -> column indices into canonical role order. Drops
        # groups with no present roles (e.g. Toe on the 6-DoF toe-ablation variant).
        self._power_groups = [
            (label, [self._roles.index(r) for r in roles if r in self._roles], rgba)
            for label, roles, rgba in _POWER_GROUPS
        ]
        self._power_groups = [g for g in self._power_groups if g[1]]

        # No CPU role_order cache needed — update() reindexes on GPU via
        # ``act._role_order`` (long tensor on the actuator device) so the
        # USD->canonical permutation happens before the single CPU transfer.

        self._step_dt = float(env.unwrapped.step_dt)
        self._step_count = 0

        # Ring buffers (canonical-order columns).
        self._capacity = int(capacity)
        self._time_buf = np.zeros(self._capacity, dtype=np.float32)
        self._buf: dict[tuple[str, str], np.ndarray] = {
            (side, sig): np.zeros((self._capacity, self._n_roles), dtype=np.float32)
            for side in ("L", "R")
            for sig in _SIGNALS
        }
        self._head = 0  # next write index
        self._count = 0  # number of valid samples (≤ capacity)
        self._lock = threading.Lock()

        # GUI thread state.
        self._sel = {key: 0 for key, *_ in _SUBPLOTS}  # selected role index per subplot
        self._power_sel = 2  # Power-tab leg selector: 0=L, 1=R, 2=L+R
        self._gui_thread: threading.Thread | None = None
        self._gui_alive = threading.Event()

        # GRF tab plumbing. Reuses the canonical heel_grf_magnitude from the tasks
        # package (same definition as the obs / heel_grf_l1 reward) by handing it a
        # SceneEntityCfg that we resolve once here — preserve_order=True pins the L/R
        # column order. Disabled gracefully if the env lacks the sensor or heel bodies.
        self._heel = None
        try:
            from isaaclab.managers import SceneEntityCfg  # noqa: PLC0415
            from hyperleg_rl.tasks.manager_based.locomotion.velocity.hyperleg.mdp import (  # noqa: PLC0415
                heel_grf_magnitude,
            )

            cfg = SceneEntityCfg("contact_forces", body_names=["l_heel", "r_heel"], preserve_order=True)
            cfg.resolve(env.unwrapped.scene)
            self._heel = {"env": env.unwrapped, "cfg": cfg, "fn": heel_grf_magnitude}
            self._grf_buf = {
                "L": np.zeros(self._capacity, dtype=np.float32),
                "R": np.zeros(self._capacity, dtype=np.float32),
            }
            self._grf_peak = {"L": 0.0, "R": 0.0}
            self._grf_peak_t = {"L": 0.0, "R": 0.0}
        except Exception as exc:
            print(f"[telemetry] heel GRF unavailable, GRF tab disabled: {exc!r}")

    # ------------------------------------------------------------------
    # Sim-thread API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Spawn the GUI thread. Idempotent."""
        if self._gui_thread is not None and self._gui_thread.is_alive():
            return
        self._gui_alive.set()
        self._gui_thread = threading.Thread(
            target=self._run_gui, daemon=True, name="HyperLegTelemetryGUI"
        )
        self._gui_thread.start()

    def stop(self) -> None:
        """Signal the GUI to close. The thread is daemon so play exit also kills it."""
        self._gui_alive.clear()

    def update(self) -> None:
        """Append one sample from env-0. Call once per ``env.step()``.

        Optimization: every GPU tensor read this step is concatenated on-device
        and transferred to the CPU in a SINGLE ``.cpu()`` call. This collapses
        what was 9 implicit ``cudaStreamSynchronize`` barriers (8 actuator
        reads + 1 GRF) into 1 — restoring ~250 us of slack per step in the
        ``--real-time`` budget so telemetry no longer causes intermittent
        hitches. The USD->canonical reindex for ``applied_effort`` happens on
        the GPU (``index_select``) before the copy, so the buffer layout and
        ``snap`` dict keys are unchanged.
        """
        env_i = self._env_index
        n = self._n_roles

        # --- Stage every GPU read on-device; no syncs yet ---------------------
        gpu_chunks: list[torch.Tensor] = []
        side_order: list[str] = []
        for side, act in self._actuators.items():
            side_order.append(side)
            # applied_effort is USD-ordered → reindex to canonical on GPU.
            tau_canon = act.applied_effort[env_i].index_select(0, act._role_order)
            gpu_chunks.append(tau_canon)                  # canonical applied_tau
            gpu_chunks.append(act.motor_heat[env_i])      # canonical motor_heat
            gpu_chunks.append(act.last_motor_tau[env_i])  # canonical motor tau
            gpu_chunks.append(act.last_motor_vel[env_i])  # canonical motor vel

        if self._heel is not None:
            grf = self._heel["fn"](self._heel["env"], self._heel["cfg"])  # (E, 2)
            gpu_chunks.append(grf[env_i].reshape(-1))     # [L, R]

        # --- SINGLE GPU->CPU synchronization ---------------------------------
        batched = (
            torch.cat(gpu_chunks, dim=0)
            .detach()
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )

        # --- Unpack on CPU (no further syncs) --------------------------------
        snap: dict[tuple[str, str], np.ndarray] = {}
        off = 0
        for side in side_order:
            tau_canon = batched[off:off + n]; off += n
            heat      = batched[off:off + n]; off += n
            tau_m     = batched[off:off + n]; off += n
            vel_m     = batched[off:off + n]; off += n
            snap[(side, "applied_tau")] = tau_canon
            snap[(side, "motor_heat")]  = heat
            # Per-joint power [W], canonical order — mirrors rewards.py
            # _compute_cot_components *before* the per-leg sum.
            # Step B+: raw τ_m for Joule (I²R proxy) and mech (τ·ω).
            p = tau_m * vel_m
            snap[(side, "joule")] = (self._power_coef * tau_m * tau_m).astype(np.float32, copy=False)
            # eta_regen=0.8, eta_out=0.9 — paper-faithful viz scaling.
            snap[(side, "mech")]  = np.where(p < 0.0, p * 0.8, p / 0.9).astype(np.float32, copy=False)

        g_L = g_R = 0.0
        if self._heel is not None:
            g_L = float(batched[off])      # preserve_order=True → col 0 = l_heel
            g_R = float(batched[off + 1])  #                       col 1 = r_heel

        t = self._step_count * self._step_dt
        with self._lock:
            i = self._head
            self._time_buf[i] = t
            for key, arr in snap.items():
                self._buf[key][i] = arr
            if self._heel is not None:
                self._grf_buf["L"][i] = g_L
                self._grf_buf["R"][i] = g_R
                if g_L > self._grf_peak["L"]:
                    self._grf_peak["L"] = g_L
                    self._grf_peak_t["L"] = t
                if g_R > self._grf_peak["R"]:
                    self._grf_peak["R"] = g_R
                    self._grf_peak_t["R"] = t
            self._head = (i + 1) % self._capacity
            if self._count < self._capacity:
                self._count += 1
        self._step_count += 1

    # ------------------------------------------------------------------
    # GUI-thread internals
    # ------------------------------------------------------------------
    def _snapshot(self) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray], dict | None]:
        """Copy buffers in chronological order under the lock."""
        with self._lock:
            count = self._count
            head = self._head
            cap = self._capacity
            if count == 0:
                empty = np.zeros(0, dtype=np.float32)
                grf = None
                if self._heel is not None:
                    grf = {"L": empty.copy(), "R": empty.copy(),
                           "peak_L": (0.0, 0.0), "peak_R": (0.0, 0.0)}
                return empty, {k: np.zeros((0, self._n_roles), dtype=np.float32) for k in self._buf}, grf
            if count < cap:
                xs = self._time_buf[:count].copy()
                snap = {k: v[:count].copy() for k, v in self._buf.items()}
                grf_L = self._grf_buf["L"][:count].copy() if self._heel is not None else None
                grf_R = self._grf_buf["R"][:count].copy() if self._heel is not None else None
            else:
                xs = np.concatenate([self._time_buf[head:], self._time_buf[:head]])
                snap = {
                    k: np.concatenate([v[head:], v[:head]], axis=0)
                    for k, v in self._buf.items()
                }
                if self._heel is not None:
                    grf_L = np.concatenate([self._grf_buf["L"][head:], self._grf_buf["L"][:head]])
                    grf_R = np.concatenate([self._grf_buf["R"][head:], self._grf_buf["R"][:head]])
                else:
                    grf_L = grf_R = None
            grf = None
            if self._heel is not None:
                grf = {
                    "L": grf_L, "R": grf_R,
                    "peak_L": (self._grf_peak["L"], self._grf_peak_t["L"]),
                    "peak_R": (self._grf_peak["R"], self._grf_peak_t["R"]),
                }
        return xs, snap, grf

    def _run_gui(self) -> None:
        """Daemon-thread entry point. Blocks inside immapp.run until window closes."""
        try:
            from imgui_bundle import hello_imgui, immapp
        except ImportError as exc:  # pragma: no cover
            print(f"[telemetry] imgui_bundle missing, GUI disabled: {exc}")
            self._gui_alive.clear()
            return

        runner_params = hello_imgui.RunnerParams()
        runner_params.app_window_params.window_title = "HyperLeg Telemetry (env 0)"
        runner_params.app_window_params.window_geometry.size = (960, 760)
        runner_params.callbacks.show_gui = self._draw_frame

        addons = immapp.AddOnsParams()
        addons.with_implot = True

        try:
            immapp.run(runner_params=runner_params, add_ons_params=addons)
        except Exception as exc:  # pragma: no cover  — GLFW errors are platform-dependent
            print(f"[telemetry] GUI thread exited: {exc!r}")
        finally:
            self._gui_alive.clear()

    def _draw_frame(self) -> None:
        from imgui_bundle import imgui

        xs, snap, grf = self._snapshot()

        imgui.text(
            f"env {self._env_index}   t = {xs[-1]:.2f} s   samples = {len(xs)}/{self._capacity}"
            if len(xs)
            else "Waiting for first sample..."
        )
        imgui.separator()

        if imgui.begin_tab_bar("telemetry_tabs"):
            if imgui.begin_tab_item_simple("Motors"):
                self._draw_motors_tab(xs, snap)
                imgui.end_tab_item()
            if imgui.begin_tab_item_simple("Power"):
                self._draw_power_tab(xs, snap)
                imgui.end_tab_item()
            if imgui.begin_tab_item_simple("GRF"):
                self._draw_grf_tab(xs, grf)
                imgui.end_tab_item()
            imgui.end_tab_bar()

    def _draw_motors_tab(
        self, xs: np.ndarray, snap: dict[tuple[str, str], np.ndarray]
    ) -> None:
        from imgui_bundle import imgui

        avail_h = imgui.get_content_region_avail().y
        plot_h = max(160.0, (avail_h - 70.0) / 2.0)  # two stacked subplots, each with a combo
        for key, signal, title, ylabel in _SUBPLOTS:
            self._draw_subplot(key, signal, title, ylabel, xs, snap, plot_h)

    def _draw_power_tab(
        self, xs: np.ndarray, snap: dict[tuple[str, str], np.ndarray]
    ) -> None:
        from imgui_bundle import imgui, implot

        imgui.text("leg")
        imgui.same_line()
        imgui.set_next_item_width(120.0)
        changed, sel = imgui.combo("##power_leg", self._power_sel, ["L", "R", "L+R"])
        if changed:
            self._power_sel = sel
        sides = ("L",) if self._power_sel == 0 else ("R",) if self._power_sel == 1 else ("L", "R")

        avail_h = imgui.get_content_region_avail().y
        plot_h = max(160.0, (avail_h - 50.0) / 2.0)
        af = implot.AxisFlags_.auto_fit.value

        # Subplot 1 — power per joint GROUP (Joule + Mech), one summed line each:
        # Hip(HY+HR+HP) / Knee / Ankle(AK+FT) / Toe.
        imgui.text("Joint-group power consumption  (Joule heating + Mech work)")
        if implot.begin_plot("##power_joints", imgui.ImVec2(-1.0, plot_h)):
            implot.setup_axes("t [s]", "Power [W]", af, af)
            if len(xs) >= 2:
                per_joint = sum(snap[(s, "joule")] + snap[(s, "mech")] for s in sides)
                for label, cols, rgba in self._power_groups:
                    ys = np.ascontiguousarray(per_joint[:, cols].sum(axis=1))
                    implot.set_next_line_style(imgui.ImVec4(*rgba), 1.5)
                    implot.plot_line(label, xs, ys)
            implot.end_plot()

        # Subplot 2 — Joule heating (red, always >=0) / Mechanical power (blue, signed:
        # +ve motoring in Q1/Q3, -ve regen in Q2/Q4) / Total = sum (green fill).
        imgui.text("Power breakdown  (Joule heating / Mechanical power / Total)")
        if implot.begin_plot("##power_breakdown", imgui.ImVec2(-1.0, plot_h)):
            implot.setup_axes("t [s]", "Power [W]", af, af)
            if len(xs) >= 2:
                joule_tot = np.ascontiguousarray(sum(snap[(s, "joule")].sum(axis=1) for s in sides))
                mech_tot = np.ascontiguousarray(sum(snap[(s, "mech")].sum(axis=1) for s in sides))
                total = np.ascontiguousarray(joule_tot + mech_tot)
                implot.set_next_fill_style(imgui.ImVec4(0.55, 0.95, 0.45, 1.0), 0.45)
                implot.plot_shaded("Total", xs, total, 0.0)
                implot.set_next_line_style(imgui.ImVec4(1.00, 0.30, 0.25, 1.0), 1.5)
                implot.plot_line("Joule heating", xs, joule_tot)
                implot.set_next_line_style(imgui.ImVec4(0.30, 0.55, 1.00, 1.0), 1.5)
                implot.plot_line("Mechanical power", xs, mech_tot)
            implot.end_plot()

    def _draw_grf_tab(self, xs: np.ndarray, grf: dict | None) -> None:
        from imgui_bundle import imgui, implot

        if grf is None:
            imgui.text_unformatted(
                "Heel GRF unavailable on this env (no 'contact_forces' sensor or no '*_heel' bodies)."
            )
            return

        cur_L = float(grf["L"][-1]) if grf["L"].size else 0.0
        cur_R = float(grf["R"][-1]) if grf["R"].size else 0.0
        pk_L, t_L = grf["peak_L"]
        pk_R, t_R = grf["peak_R"]
        t_now = float(xs[-1]) if len(xs) else 0.0
        # Flash the peak line for ~1 s after it was just set.
        fresh_L = pk_L > 0.0 and (t_now - t_L) < 1.0
        fresh_R = pk_R > 0.0 and (t_now - t_R) < 1.0
        # Shared gauge denominator so the two bars are visually comparable.
        gauge_max = max(pk_L, pk_R, 1.0)

        blue = imgui.ImVec4(0.20, 0.55, 1.00, 1.00)  # L (matches Motors-tab convention)
        red = imgui.ImVec4(1.00, 0.38, 0.30, 1.00)   # R
        gold = imgui.ImVec4(1.00, 0.85, 0.20, 1.00)  # fresh-peak flash
        dim = imgui.ImVec4(0.80, 0.80, 0.80, 1.00)

        pane_w = imgui.get_content_region_avail().x / 2.0 - 8.0
        bar_h = 20.0

        # ---- L pane ----
        imgui.begin_group()
        imgui.text_colored(blue, "L heel")
        imgui.text(f"current : {cur_L:8.1f} N")
        imgui.text_colored(gold if fresh_L else dim, f"peak    : {pk_L:8.1f} N   (t = {t_L:6.2f} s)")
        imgui.progress_bar(min(cur_L / gauge_max, 1.0), imgui.ImVec2(pane_w, bar_h), f"{cur_L:.0f} N")
        imgui.end_group()

        imgui.same_line(0.0, 16.0)

        # ---- R pane ----
        imgui.begin_group()
        imgui.text_colored(red, "R heel")
        imgui.text(f"current : {cur_R:8.1f} N")
        imgui.text_colored(gold if fresh_R else dim, f"peak    : {pk_R:8.1f} N   (t = {t_R:6.2f} s)")
        imgui.progress_bar(min(cur_R / gauge_max, 1.0), imgui.ImVec2(pane_w, bar_h), f"{cur_R:.0f} N")
        imgui.end_group()

        # ---- Reset peaks button ----
        if imgui.button("Reset peaks"):
            with self._lock:
                self._grf_peak["L"] = 0.0
                self._grf_peak["R"] = 0.0
                self._grf_peak_t["L"] = 0.0
                self._grf_peak_t["R"] = 0.0

        # ---- Time series ----
        avail_h = imgui.get_content_region_avail().y
        plot_h = max(180.0, avail_h - 16.0)
        af = implot.AxisFlags_.auto_fit.value
        if implot.begin_plot("##grf_ts", imgui.ImVec2(-1.0, plot_h)):
            implot.setup_axes("t [s]", "|F_heel| [N]", af, af)
            if len(xs) >= 2:
                implot.set_next_line_style(blue, 1.5)
                implot.plot_line("L heel", xs, np.ascontiguousarray(grf["L"]))
                implot.set_next_line_style(red, 1.5)
                implot.plot_line("R heel", xs, np.ascontiguousarray(grf["R"]))
            implot.end_plot()

    def _draw_subplot(
        self,
        key: str,
        signal: str,
        title: str,
        ylabel: str,
        xs: np.ndarray,
        snap: dict[tuple[str, str], np.ndarray],
        plot_h: float,
    ) -> None:
        from imgui_bundle import imgui, implot

        imgui.text(title)
        imgui.same_line()
        imgui.set_next_item_width(120.0)
        sel = self._sel[key]
        changed, sel_new = imgui.combo(f"role##{key}", sel, self._roles)
        if changed:
            self._sel[key] = sel_new
            sel = sel_new

        if implot.begin_plot(f"##plot_{key}", imgui.ImVec2(-1.0, plot_h)):
            # auto_fit on both axes → axes track the latest buffer every frame.
            af = implot.AxisFlags_.auto_fit.value
            implot.setup_axes("t [s]", ylabel, af, af)
            if len(xs) >= 2:
                role_label = self._roles[sel]
                # L: blue, R: red. Distinct colors give clear bilateral readout.
                colors = {
                    "L": imgui.ImVec4(0.20, 0.55, 1.00, 1.00),
                    "R": imgui.ImVec4(1.00, 0.38, 0.30, 1.00),
                }
                for side in ("L", "R"):
                    # Column slice of a 2D array is non-contiguous; implot needs 1D contiguous.
                    ys = np.ascontiguousarray(snap[(side, signal)][:, sel])
                    implot.set_next_line_style(colors[side], 1.5)
                    implot.plot_line(f"{side}_{role_label}", xs, ys)
            implot.end_plot()
