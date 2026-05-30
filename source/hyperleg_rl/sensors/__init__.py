# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom sensors for HyperLeg."""

from .force_vector_contact_sensor import (
    ForceVectorContactSensor,
    ForceVectorContactSensorCfg,
)

__all__ = ["ForceVectorContactSensor", "ForceVectorContactSensorCfg"]
