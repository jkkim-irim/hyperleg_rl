# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators import DCMotorCfg

from .actuator_pd import IdentifiedActuator


@configclass
class IdentifiedActuatorCfg(DCMotorCfg):
    """Configuration for direct control (DC) motor actuator model."""

    class_type: type = IdentifiedActuator

    friction_coulomb: float = MISSING
    """ (in N-m)."""
    activation_vel: float = MISSING
    """ (in Rad/s)."""
    friction_viscous: float = MISSING
    """ (in N-m-s/Rad)."""


from omni.isaac.lab.actuators.actuator_cfg import ActuatorBaseCfg
from .actuator_pd import IRIMActuator

@configclass
class IRIMActuatorCfg(ActuatorBaseCfg):
    class_type: type = IRIMActuator



    friction_coulomb: dict[str, float] | float = MISSING
    """ (in N-m)."""
    activation_vel: dict[str, float] | float = MISSING
    """ (in Rad/s)."""
    friction_viscous: dict[str, float] | float = MISSING
    """ (in N-m-s/Rad)."""


    # actuatoer_effort_limit: dict[str, float] | float = MISSING
    # actuatoer_velocity_limit: dict[str, float] | float = MISSING


