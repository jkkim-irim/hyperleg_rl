# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HyperLeg biped robot config (HyperLeg.usd).

14-DoF biped: 7 joints per leg (Hip Yaw / Hip Roll / Hip Pitch / Knee /
Ankle Pitch / Ankle Roll / Toe). Joint and body names match
``projects/assets/hyperleg/HyperLeg.xml``.

The shipped ``HyperLeg.usd`` was patched once via
``scripts/fix_usd_inertia.py`` to populate ``UsdPhysics.MassAPI``
(mass / diagonal inertia / CoM / principal axes) from the MJCF
``<inertial>`` blocks. After patching, the USD is loaded directly via
:class:`UsdFileCfg`.

Actuation uses one :class:`CoupledLegActuator` per leg, which implements
the paper's Cooperative-Actuation model (§II-B, eq. 1-5): a constant 7×7
Jacobian per leg maps joint torques/velocities to motor space, a
4-quadrant DC motor saturation is applied at the motor stage, and joint-space
coulomb·tanh + viscous friction (Table I) is subtracted from the result.
See ``dvcc/02_actuator_irim.md`` for derivation and design notes.
"""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from hyperleg_rl.actuators import CoupledLegActuatorCfg

# {repo_root}/assets/hyperleg/HyperLeg.usd
_REPO_ROOT = Path(__file__).resolve().parents[3]
_HYPERLEG_USD_PATH = _REPO_ROOT / "assets" / "hyperleg" / "HyperLeg.usd"


# Per-leg Jacobian J = ∂motor/∂joint, canonical order [HY HR HP KN AK FT TO].
# Hip block: diag(25)  — 25:1 cycloid, decoupled.
# Lower-leg 4×4: standing-pose linearization, paper eq. (3) lower-right
# block (transposed; equivalent to legacy `actuator_pd.py::jacob_inv`).
_J_PER_LEG = (
    (25.0,  0.0,  0.0,   0.0,    0.0,    0.0,    0.0  ),
    ( 0.0, 25.0,  0.0,   0.0,    0.0,    0.0,    0.0  ),
    ( 0.0,  0.0, 25.0,   0.0,    0.0,    0.0,    0.0  ),
    ( 0.0,  0.0,  0.0, -31.11,   0.0,    0.0,    0.0  ),
    ( 0.0,  0.0,  0.0, -31.11,  26.69,  16.51,   0.0  ),
    ( 0.0,  0.0,  0.0, -31.11,  26.69, -16.51,   0.0  ),
    ( 0.0,  0.0,  0.0, -31.11,  19.23,   0.0,   21.27),
)


def _leg_actuator_cfg(side: str) -> CoupledLegActuatorCfg:
    """One ``CoupledLegActuatorCfg`` covering all 7 joints of side L or R."""
    p = f"{side}_"  # "L_" or "R_"
    return CoupledLegActuatorCfg(
        joint_names_expr=[
            f"{p}HY", f"{p}HR", f"{p}HP",
            f"{p}KN", f"{p}AK", f"{p}FT", f"{p}TO",
        ],
        # Joint-space PD (paper eq. 4 form)
        stiffness={
            f"{p}HY": 50.0, f"{p}HR": 50.0, f"{p}HP": 50.0,
            f"{p}KN": 100.0, f"{p}AK": 30.0,
            f"{p}FT": 20.0, f"{p}TO": 15.0,
        },
        damping={
            f"{p}HY": 4.0, f"{p}HR": 5.0, f"{p}HP": 5.0,
            f"{p}KN": 10.0, f"{p}AK": 3.0,
            f"{p}FT": 2.0, f"{p}TO": 1.0,
        },
        # Joint-space armature (PhysX adds to mass-matrix diagonal). Uniform 0.01
        # as a stability baseline; refine per-joint with γ²·I_rotor if needed.
        armature={
            f"{p}HY": 0.01, f"{p}HR": 0.01, f"{p}HP": 0.01,
            f"{p}KN": 0.01, f"{p}AK": 0.01,
            f"{p}FT": 0.01, f"{p}TO": 0.01,
        },
        # Joint-side coulomb + viscous friction (paper Table I, eq. 5)
        friction_coulomb={
            f"{p}HY": 3.30, f"{p}HR": 3.30, f"{p}HP": 3.30,
            f"{p}KN": 5.59, f"{p}AK": 2.09,
            f"{p}FT": 2.20, f"{p}TO": 0.88,
        },
        friction_viscous={
            f"{p}HY": 0.08, f"{p}HR": 0.08, f"{p}HP": 0.08,
            f"{p}KN": 0.10, f"{p}AK": 0.10,
            f"{p}FT": 0.10, f"{p}TO": 0.10,
        },
        activation_vel=0.1,
        # Cooperative Actuation Jacobian (canonical [HY HR HP KN AK FT TO])
        jacobian_joint_to_motor=_J_PER_LEG,
        # Motor-side continuous torque [Nm] (τ_cont on the DC torque-speed envelope).
        # Typical DC rating: τ_cont ≈ τ_stall/2 (conservative: /3). KN/ankles follow the same rule.
        motor_effort_limit={
            f"{p}HY": 2.52, f"{p}HR": 2.52, f"{p}HP": 2.52,  # 5.04/2
            f"{p}KN": 2.68,
            f"{p}AK": 1.63, f"{p}FT": 1.63, f"{p}TO": 1.63,
        },
        # Motor-side stall (peak) torque [Nm] at ω=0 (τ_stall on the envelope).
        # Hip cycloid 25:1 → 5.04 Nm motor ≈ 126 Nm joint peak; τ_cont above → ~63 Nm joint sustained.
        # Knee: 212/31.11 = 6.81. Ankles: 111.7/(2·26.69) → 2.09 per motor. Toe: 46.8/21.27 = 2.20.
        motor_saturation_effort={
            f"{p}HY": 5.04, f"{p}HR": 5.04, f"{p}HP": 5.04,
            f"{p}KN": 6.81,
            f"{p}AK": 2.09, f"{p}FT": 2.09, f"{p}TO": 2.20,
        },
        motor_velocity_limit={
            # No-load speed deg/s → rad/s
            f"{p}HY": 300.0, f"{p}HR": 300.0, f"{p}HP": 300.0,  # 17,189 deg/s
            f"{p}KN": 327.2,                                     # 18,750 deg/s
            f"{p}AK": 366.0, f"{p}FT": 366.0, f"{p}TO": 366.0,   # 20,970 deg/s
        },
    )


HYPERLEG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_HYPERLEG_USD_PATH.resolve()),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.94),
        joint_pos={
            ".*_HP": math.radians(-10.0),
            ".*_KN": math.radians(20.0),
            ".*_AK": math.radians(9.5),
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "left_leg": _leg_actuator_cfg("L"),
        "right_leg": _leg_actuator_cfg("R"),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""HyperLeg biped articulation config (UsdFileCfg + per-leg CoupledLegActuator)."""
