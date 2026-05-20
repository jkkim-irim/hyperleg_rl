# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HyperLeg custom actuators.

`CoupledLegActuator` implements the Cooperative Actuation (CA) torque model
from the HyperLeg paper (§II-B, eq. 1-5): a constant 7×7 Jacobian per leg maps
joint torques to/from motor space, where a 4-quadrant DC motor saturation is
applied; joint-space coulomb·tanh + viscous friction is then subtracted.
"""

from .coupled_leg import CoupledLegActuator
from .coupled_leg_cfg import CoupledLegActuatorCfg

__all__ = ["CoupledLegActuator", "CoupledLegActuatorCfg"]
