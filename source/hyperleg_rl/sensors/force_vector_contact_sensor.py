# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""ContactSensor variant that draws net contact-force vectors as debug lines.

Uses ``isaacsim.util.debug_draw`` to render an arrow per tracked body from the
body origin along its net contact force (world frame). Toggled on/off via the
standard ``debug_vis`` flag, so the viewport adds a checkbox just like for
``height_scanner`` / ``contact_forces``.
"""

from __future__ import annotations

from typing import Any

import torch

from isaaclab.sensors.contact_sensor import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass


class ForceVectorContactSensor(ContactSensor):
    """ContactSensor that visualizes per-body net force vectors as lines.

    On each post-update tick (while ``debug_vis`` is enabled), draws a line
    from each tracked body's world position along the net contact force in
    world frame, scaled by :attr:`ForceVectorContactSensorCfg.force_scale`.
    Bodies whose force magnitude is below
    :attr:`ForceVectorContactSensorCfg.force_threshold_n` are skipped.
    """

    cfg: "ForceVectorContactSensorCfg"

    def __init__(self, cfg: "ForceVectorContactSensorCfg", *args: Any, **kwargs: Any):
        # Set before super().__init__: SensorBase.__init__ calls set_debug_vis()
        # which routes into our _set_debug_vis_impl before our body runs.
        self._draw_iface = None
        super().__init__(cfg, *args, **kwargs)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if self._draw_iface is None:
                try:
                    from isaacsim.util.debug_draw import _debug_draw  # noqa: PLC0415
                except ImportError:
                    return
                self._draw_iface = _debug_draw.acquire_debug_draw_interface()
        else:
            if self._draw_iface is not None:
                self._draw_iface.clear_lines()
                self._draw_iface.clear_points()

    def _debug_vis_callback(self, event):
        if self._draw_iface is None or self.body_physx_view is None:
            return
        forces = self._data.net_forces_w
        if forces is None:
            return
        if self.cfg.track_pose:
            positions = self._data.pos_w
        else:
            pose = self.body_physx_view.get_transforms()
            positions = pose.view(-1, self._num_bodies, 7)[:, :, :3]

        pos_flat = positions.reshape(-1, 3)
        force_flat = forces.reshape(-1, 3)
        mags = torch.linalg.norm(force_flat, dim=-1)
        mask = mags > self.cfg.force_threshold_n

        self._draw_iface.clear_lines()
        self._draw_iface.clear_points()
        if not bool(mask.any()):
            return

        starts_t = pos_flat[mask].detach().cpu().numpy()
        ends_t = (pos_flat[mask] + force_flat[mask] * self.cfg.force_scale).detach().cpu().numpy()
        n = int(starts_t.shape[0])
        starts = [tuple(map(float, p)) for p in starts_t]
        ends = [tuple(map(float, p)) for p in ends_t]
        colors = [self.cfg.line_color] * n
        sizes = [self.cfg.line_width] * n
        self._draw_iface.draw_lines(starts, ends, colors, sizes)
        self._draw_iface.draw_points(ends, [self.cfg.tip_color] * n, [self.cfg.tip_size] * n)


@configclass
class ForceVectorContactSensorCfg(ContactSensorCfg):
    """Config for :class:`ForceVectorContactSensor`."""

    class_type: type = ForceVectorContactSensor

    force_scale: float = 0.001
    """Line length per Newton (world frame)."""

    force_threshold_n: float = 0.1
    """Skip lines whose net force magnitude (N) is below this threshold."""

    line_width: float = 6.0
    """Line thickness in pixels."""

    line_color: tuple[float, float, float, float] = (1.0, 0.4, 0.0, 1.0)
    """RGBA line color."""

    tip_color: tuple[float, float, float, float] = (1.0, 1.0, 0.0, 1.0)
    """RGBA color for the tip dot."""

    tip_size: float = 8.0
    """Tip dot size in pixels."""
