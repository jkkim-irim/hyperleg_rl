# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Live telemetry plotter for HyperLeg motors (Play mode only).

Three stacked ImPlot subplots (joint-side torque / motor_heat / motor_load_avg).
Each subplot has its own role selector (HY/HR/HP/KN/AK/FT/TO); both L and R
legs of that role are drawn together so bilateral asymmetry is visible.

Designed to not bottleneck the sim main loop:
- Per env.step() the sim thread copies one env-row of three tensors per leg to
  CPU (≈ 168 B/step for 7-joint variant) and appends to numpy ring buffers
  under a short-held lock.
- A daemon thread runs the GLFW/ImPlot window via ``imgui_bundle.immapp.run``;
  it copies a snapshot under the lock then renders without holding it.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hyperleg_rl.actuators.coupled_leg import CoupledLegActuator


# Canonical role order matches CoupledLegActuator._ROLE_SUFFIXES.
_ALL_ROLES: tuple[str, ...] = ("HY", "HR", "HP", "KN", "AK", "FT", "TO")
_SIGNALS: tuple[str, ...] = ("applied_tau", "motor_heat", "motor_load_avg")

# Subplot configuration: (key, signal, title, y-axis label).
_SUBPLOTS: tuple[tuple[str, str, str, str], ...] = (
    ("top", "applied_tau", "Joint-side torque (applied_effort)", "tau_j [Nm]"),
    ("mid", "motor_heat", "Motor heat EMA  (tau_m / tau_cont)^2", "heat [-]"),
    ("bot", "motor_load_avg", "Motor load EMA  |tau_m / tau_cont|", "load [-]"),
)


class MotorTelemetryPlotter:
    """Live ImPlot window showing per-joint motor telemetry for env 0.

    Args:
        env: A wrapped Isaac Lab env. ``env.unwrapped.scene["robot"].actuators``
            must contain ``"left_leg"`` and ``"right_leg"`` keyed
            ``CoupledLegActuator`` instances.
        capacity: Ring-buffer length. Default 500 ≈ 10 s at 50 Hz control rate.
        env_index: Which env to plot. Defaults to 0.
    """

    def __init__(self, env, *, capacity: int = 500, env_index: int = 0) -> None:
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

        # Cache role_order (canonical->USD) per side as CPU int64 arrays.
        self._role_order = {
            side: act._role_order.detach().cpu().numpy().astype(np.int64)
            for side, act in self._actuators.items()
        }

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
        self._gui_thread: threading.Thread | None = None
        self._gui_alive = threading.Event()

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
        """Append one sample from env-0. Call once per ``env.step()``."""
        snap: dict[tuple[str, str], np.ndarray] = {}
        env_i = self._env_index
        for side, act in self._actuators.items():
            ro = self._role_order[side]
            # applied_effort is USD-ordered → reindex into canonical.
            tau_usd = act.applied_effort[env_i].detach().cpu().numpy()
            snap[(side, "applied_tau")] = tau_usd[ro].astype(np.float32, copy=False)
            # motor_heat / motor_load_avg are already canonical (see coupled_leg.py L106-116).
            snap[(side, "motor_heat")] = (
                act.motor_heat[env_i].detach().cpu().numpy().astype(np.float32, copy=False)
            )
            snap[(side, "motor_load_avg")] = (
                act.motor_load_avg[env_i].detach().cpu().numpy().astype(np.float32, copy=False)
            )

        t = self._step_count * self._step_dt
        with self._lock:
            i = self._head
            self._time_buf[i] = t
            for key, arr in snap.items():
                self._buf[key][i] = arr
            self._head = (i + 1) % self._capacity
            if self._count < self._capacity:
                self._count += 1
        self._step_count += 1

    # ------------------------------------------------------------------
    # GUI-thread internals
    # ------------------------------------------------------------------
    def _snapshot(self) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray]]:
        """Copy buffers in chronological order under the lock."""
        with self._lock:
            count = self._count
            head = self._head
            cap = self._capacity
            if count == 0:
                empty = np.zeros(0, dtype=np.float32)
                return empty, {k: np.zeros((0, self._n_roles), dtype=np.float32) for k in self._buf}
            if count < cap:
                xs = self._time_buf[:count].copy()
                snap = {k: v[:count].copy() for k, v in self._buf.items()}
            else:
                xs = np.concatenate([self._time_buf[head:], self._time_buf[:head]])
                snap = {
                    k: np.concatenate([v[head:], v[:head]], axis=0)
                    for k, v in self._buf.items()
                }
        return xs, snap

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

        xs, snap = self._snapshot()

        imgui.text(
            f"env {self._env_index}   t = {xs[-1]:.2f} s   samples = {len(xs)}/{self._capacity}"
            if len(xs)
            else "Waiting for first sample..."
        )
        imgui.separator()

        avail_h = imgui.get_content_region_avail().y
        # Roughly equal thirds, leaving room for each combobox + label.
        plot_h = max(140.0, (avail_h - 90.0) / 3.0)

        for key, signal, title, ylabel in _SUBPLOTS:
            self._draw_subplot(key, signal, title, ylabel, xs, snap, plot_h)

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
            implot.setup_axes("t [s]", ylabel)
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
