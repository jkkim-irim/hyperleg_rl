# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
from berkeley_humanoid.actuators import IdentifiedActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

from berkeley_humanoid.assets import ISAAC_ASSET_DIR



from berkeley_humanoid.actuators.actuator_cfg import IRIMActuatorCfg

HYPERLEG_L_COUPLED_ACTUATOR_CFG = IRIMActuatorCfg(
    # 이 정규식은 "L_" 다음에 KN|AK|FT|TO 중 하나가 붙는 모든 조인트를 매칭
    joint_names_expr=[r".*L_(KN|AK|FT|TO)"],
    
    effort_limit={"L_KN": 2.68, "L_AK": 1.63, "L_FT": 1.63, "L_TO": 1.63},
    velocity_limit={"L_KN": 327.2492, "L_AK": 365.9955, "L_FT": 365.9955, "L_TO": 365.9955},

    # KN, AK, FT, TO 순서대로 stiffness, damping을 텐서로
    stiffness={"L_KN": 63.0, "L_AK": 35.0, "L_FT": 30.0, "L_TO": 15.0},
    damping={"L_KN": 10.0, "L_AK": 4.0, "L_FT": 3.5, "L_TO": 3.5},

    armature={
        "L_KN": (2.280e-5 * pow(31.11, 2)) + (8.048e-6 * pow(31.11, 2))*3,
        "L_AK": (8.048e-6 * pow(26.69, 2))*2 + (8.048e-6 * pow(19.23, 2)),
        "L_FT": (8.048e-6 * pow(16.51, 2))*2,
        "L_TO": 8.048e-6 * pow(21.27, 2)
    },

    friction_coulomb={"L_KN": 5.59, "L_AK": 2.09, "L_FT": 2.2, "L_TO": 0.88},
    activation_vel={"L_KN": 0.1, "L_AK": 0.1, "L_FT": 0.1, "L_TO": 0.1},
    friction_viscous={"L_KN": 0.1, "L_AK": 0.1, "L_FT": 0.1, "L_TO": 0.1},
)

HYPERLEG_R_COUPLED_ACTUATOR_CFG = IRIMActuatorCfg(
    joint_names_expr=[r".*R_(KN|AK|FT|TO)"],
    
    effort_limit={"R_KN": 2.68, "R_AK": 1.63, "R_FT": 1.63, "R_TO": 1.63},
    velocity_limit={"R_KN": 327.2492, "R_AK": 365.9955, "R_FT": 365.9955, "R_TO": 365.9955},

    # KN, AK, FT, TO 순서대로 stiffness, damping을 텐서로
    stiffness={"R_KN": 63.0, "R_AK": 35.0, "R_FT": 30.0, "R_TO": 15.0},
    damping={"R_KN": 10.0, "R_AK": 4.0, "R_FT": 3.5, "R_TO": 3.5},

    armature={
        "R_KN": (2.280e-5 * pow(31.11, 2)) + (8.048e-6 * pow(31.11, 2))*3, #0.04543
        "R_AK": (8.048e-6 * pow(26.69, 2))*2 + (8.048e-6 * pow(19.23, 2)), #0.01444
        "R_FT": (8.048e-6 * pow(16.51, 2))*2, 
        "R_TO": 8.048e-6 * pow(21.27, 2)
    },
    friction_coulomb={"R_KN": 5.59, "R_AK": 2.09, "R_FT": 2.2, "R_TO": 0.88},
    activation_vel={"R_KN": 0.1, "R_AK": 0.1, "R_FT": 0.1, "R_TO": 0.1},
    friction_viscous={"R_KN": 0.1, "R_AK": 0.1, "R_FT": 0.1, "R_TO": 0.1},
)

BERKELEY_HUMANOID_HXX_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*HY", ".*HR"],
    effort_limit=126.0,
    velocity_limit=12.0,
    saturation_effort=None,
    stiffness={".*HY": 72.0, ".*HR": 80.0},
    damping={".*HY": 8.0, ".*HR": 8.0},
    armature={".*": 2.17743954778e-4 * 625},
    friction_coulomb=3.30272,
    activation_vel=0.1,
    friction_viscous=0.08,
)

BERKELEY_HUMANOID_HP_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*HP"],
    effort_limit=126.0,
    velocity_limit=12.0,
    saturation_effort=None,
    stiffness={".*": 100.0},
    damping={".*": 10.0},
    armature={".*": 2.17743954778e-4 * 625},
    friction_coulomb=3.30272,
    activation_vel=0.1,
    friction_viscous=0.08,
)


BERKELEY_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Usd/Allex_RL_Model_4_contact_point.usd",
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
            enabled_self_collisions=True, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=2
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.94),  # 초기 위치는 0번 방에 설정
        joint_pos={

            # 왼쪽 다리
            'L_HY': 0.0,    
            'L_HR': 0.0,    
            'L_HP': -0.174533,  
            'L_KN': 0.349066,   # 20 degree
            'L_AK': 0.174533,   # 10 degree
            'L_FT': 0.0, 
            'L_TO': 0.0, 

            # 오른쪽 다리
            'R_HY': 0.0,    
            'R_HR': 0.0,    
            'R_HP': -0.174533, 
            'R_KN': 0.349066,  
            'R_AK': 0.174533, 
            'R_FT': 0.0, 
            'R_TO': 0.0, 
        }
    ),
    actuators={"hxx": BERKELEY_HUMANOID_HXX_ACTUATOR_CFG, 
               "hp": BERKELEY_HUMANOID_HP_ACTUATOR_CFG,
               "l_leg": HYPERLEG_L_COUPLED_ACTUATOR_CFG, 
               "r_leg": HYPERLEG_R_COUPLED_ACTUATOR_CFG
               },
    soft_joint_pos_limit_factor=0.95,
)