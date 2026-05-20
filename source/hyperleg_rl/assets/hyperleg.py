# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HyperLeg biped robot config (HyperLeg.usd).

14-DoF biped: 7 joints per leg (Hip Yaw / Hip Roll / Hip Pitch / Knee /
Ankle / Foot / Toe). Joint and body names match
``projects/assets/hyperleg/HyperLeg.xml``.

The shipped ``HyperLeg.usd`` was patched once via
``scripts/fix_usd_inertia.py`` to populate ``UsdPhysics.MassAPI``
(mass / diagonal inertia / CoM / principal axes) from the MJCF
``<inertial>`` blocks. After patching, the USD is loaded directly via
:class:`UsdFileCfg`.

Per-joint actuators use the :class:`DCMotorCfg` four-quadrant torque-speed
saturation model on top of explicit PD control. Joints are grouped by motor
class (hips, knees, ankles, feet/toes) because :attr:`DCMotorCfg.saturation_effort`
(stall torque) is a per-group scalar.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

# {repo_root}/assets/hyperleg/HyperLeg.usd
_REPO_ROOT = Path(__file__).resolve().parents[3]
_HYPERLEG_USD_PATH = _REPO_ROOT / "assets" / "hyperleg" / "HyperLeg.usd"


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
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(ensure_drives_exist=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.94),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hips": DCMotorCfg(
            joint_names_expr=[".*_HY", ".*_HR", ".*_HP"],
            effort_limit={".*_HY": 60.0, ".*_HR": 60.0, ".*_HP": 60.0},
            saturation_effort=126.0,
            velocity_limit=25.0,
            stiffness={".*_HY": 100.0, ".*_HR": 150.0, ".*_HP": 150.0},
            damping={".*_HY": 4.0, ".*_HR": 5.0, ".*_HP": 5.0},
        ),
        "knees": DCMotorCfg(
            joint_names_expr=[".*_KN"],
            effort_limit=100.0,
            saturation_effort=212.0,
            velocity_limit=18.0,
            stiffness=200.0,
            damping=6.0,
        ),
        "ankles": DCMotorCfg(
            joint_names_expr=[".*_AK"],
            effort_limit=50,
            saturation_effort=111.7,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=2.0,
        ),
        "foot": DCMotorCfg(
            joint_names_expr=[".*_FT"],
            effort_limit=20.0,
            saturation_effort=48.4,
            velocity_limit=35.0,
            stiffness=20.0,
            damping=1.0,
        ),
        "toe": DCMotorCfg(
            joint_names_expr=[".*_TO"],
            effort_limit=20.0,
            saturation_effort=46.8,
            velocity_limit=35.0,
            stiffness=15.0,
            damping=0.8,
        ),
    },
)
"""HyperLeg biped articulation config (UsdFileCfg + position-PD actuator)."""
