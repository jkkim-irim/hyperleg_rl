from __future__ import annotations

# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 로그 임계치 (예: 2000 스텝)
_LOG_THRESHOLD = 2000
BODY_DATA_LOG_MAX_COUNT = 2000

_ALL_JOINT_LOG = False
_LEFT_FOOT_COP = False
_RIGHT_FOOT_COP = False
_ROBOT_BODY = False

_ROBOT_BODY_DATA_PLOT_QT = False
if (_ROBOT_BODY_DATA_PLOT_QT):
    import pyqtgraph as pg
    from PyQt5 import QtCore, QtWidgets

COP_PLOT_QT = False
if (COP_PLOT_QT):
    import pyqtgraph as pg
    from PyQt5 import QtCore, QtWidgets

JOINT_PLOT_QT = False
if (JOINT_PLOT_QT):
    import pyqtgraph as pg
    from PyQt5 import QtCore, QtWidgets

import torch
import time
from typing import TYPE_CHECKING
from collections import deque
import numpy as np

from omni.isaac.core.utils.types import ArticulationActions
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.robots import RobotView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.prims import RigidContactView
from omni.isaac.core.utils.rotations import quat_to_euler_angles
from omni.isaac.core.utils.numpy import rad2deg

from omni.isaac.lab.actuators import DCMotor


if TYPE_CHECKING:



    from .actuator_cfg import IdentifiedActuatorCfg

class IdentifiedActuator(DCMotor):
    cfg: IdentifiedActuatorCfg

    def __init__(self, cfg: IdentifiedActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # coulomb 
        self.friction_coulomb = self._parse_joint_parameter(self.cfg.friction_coulomb, 0.)
        self.activation_vel = self._parse_joint_parameter(self.cfg.activation_vel, torch.inf)
        self.friction_viscous = self._parse_joint_parameter(self.cfg.friction_viscous, 0.)
        # print("Hip Friction Coulomb: ", self.friction_coulomb)
        # print("Hip Activation Vel: ", self.activation_vel)
        # print("Hip Friction viscous: ", self.friction_viscous)


        self.velocity_limit = self._parse_joint_parameter(self.cfg.velocity_limit, 0.)
        self.effort_limit = self._parse_joint_parameter(self.cfg.effort_limit, 0.)
        self._actuator_velo = torch.zeros_like(self.computed_effort)

        # print(self._joint_names)
        # print(self._joint_indices)

        # 로그 버퍼 초기화 (applied_effort와 actuator_velo)
        self._log_joint_names = list(self._joint_names)
        self._log_applied_effort = []
        self._log_actuator_velo = []

    def compute(
            self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        
        self._actuator_velo = joint_vel

        control_action = super().compute(control_action, joint_pos, joint_vel)

        # apply friction model on the torque
        control_action.joint_efforts = control_action.joint_efforts - (self.friction_coulomb * torch.tanh(
            joint_vel / self.activation_vel) + self.friction_viscous * joint_vel)

        self.applied_effort = control_action.joint_efforts

        control_action.joint_positions = None
        control_action.joint_velocities = None

        return control_action
    
    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # compute torque limits

        # 각각의 조건별 min / max 값을 미리 계산
        max_if_velo_positive = self.effort_limit * (1.0 - self._actuator_velo / self.velocity_limit)
        min_if_velo_negative = self.effort_limit * (-1.0 - self._actuator_velo / self.velocity_limit)

        # velo >= 0 이면 "양수에 대한 min/max" 값을, 그렇지 않으면 "음수에 대한 min/max" 값을 고른다.
        
        max_effort = torch.where(
            self._actuator_velo >= 0,
            max_if_velo_positive,
            self.effort_limit
        )

        min_effort = torch.where(
            self._actuator_velo >= 0,
            -self.effort_limit,
            min_if_velo_negative
        )

        return torch.clip(torch.clip(effort, min=min_effort, max=max_effort), -self.effort_limit, self.effort_limit)

import os
import csv
from dataclasses import dataclass

@dataclass
class IRIMActuatorCfg:
    velocity_limit: float
    friction_coulomb: float = 0.0
    activation_vel: float = float('inf')
    friction_viscous: float = 0.0

class LazyBool:
    def __init__(self, value: bool):
        self.func = lambda: value

from collections.abc import Sequence
from omni.isaac.lab.actuators.actuator_base import ActuatorBase
if TYPE_CHECKING:
    from .actuator_cfg import IRIMActuatorCfg
class IRIMActuator(ActuatorBase):
    
    cfg: IRIMActuatorCfg
    def __init__(self, cfg: IRIMActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # 데이터 저장소 초기화
        self.data_buffer = {
            'positions': deque(maxlen=200),
            'applied': deque(maxlen=200),
            'measured': deque(maxlen=200),
            'velocities': deque(maxlen=200)
        }

        # 모든 관절 이름 정의
        self.all_joint_names = ['L_HY', 'R_HY', 'L_HR', 'R_HR', 'L_HP', 'R_HP', 
                              'L_KN', 'R_KN', 'L_AK', 'R_AK','R_FT']
        
        self._zeros_effort = torch.zeros_like(self.computed_effort)
        self.step_counter = 0
        # check that quantities are provided
        if self.cfg.velocity_limit is None:
            raise ValueError("The velocity limit must be provided for the DC motor actuator model.")

        # form DCMotor
        self._actuator_velo = torch.zeros_like(self.computed_effort)
        self._actuator_effort = torch.zeros_like(self.computed_effort)
        self._cliped_actu_effort = torch.zeros_like(self.computed_effort)

        # form berkeley
        self.friction_coulomb = self._parse_joint_parameter(self.cfg.friction_coulomb, 0.)
        self.activation_vel = self._parse_joint_parameter(self.cfg.activation_vel, torch.inf)
        self.friction_viscous = self._parse_joint_parameter(self.cfg.friction_viscous, 0.)

        device = self.computed_effort.device if hasattr(self, 'computed_effort') else torch.device('cpu')
        self.jacob = torch.tensor([
            [-0.0321,  0.0000,  0.0000,  0.0000],
            [-0.0375,  0.0187,  0.0187,  0.0000],
            [ 0.0000,  0.0303, -0.0303,  0.0000],
            [-0.0131, -0.0169, -0.0169,  0.0470]
        ], dtype=torch.float32).to(device)

        self.jacob_T = torch.tensor([
            [-0.0321, -0.0375,  0.0000, -0.0131],
            [ 0.0000,  0.0187,  0.0303, -0.0169],
            [ 0.0000,  0.0187, -0.0303, -0.0169],
            [ 0.0000,  0.0000,  0.0000,  0.0470]
        ], dtype=torch.float32).to(device)

        self.jacob_inv = torch.tensor([
            [-31.11,   0.00,   0.00,   0.00],
            [-31.11,  26.69,  16.51,   0.00],
            [-31.11,  26.69, -16.51,   0.00],
            [-31.11,  19.23,   0.00,  21.27]
        ], dtype=torch.float32).to(device)

        self.jacob_inv_T = torch.tensor([
            [-31.11, -31.11, -31.11, -31.11],
            [     0,  26.69,  26.69,  19.23],
            [     0,  16.51, -16.51,      0],
            [     0,      0,      0,  21.27]
        ], dtype=torch.float32).to(device)

        self.prim_paths = "/World/envs/env_0/Robot"
        self.robot = RobotView(prim_paths_expr=self.prim_paths,name="Robot")
        self.robot.initialize()

        """ Joint Data 로깅을 위한 Robot Initialization """
        if (_ALL_JOINT_LOG):
            self.prim_paths = "/World/envs/env_0/Robot"
            self.robot = RobotView(prim_paths_expr=self.prim_paths,name="Robot")
            self.robot.initialize()
            self.joint_positions_log = []  # 14개 조인트 데이터 로깅을 위한 리스트 초기화
            self.joint_applied_efforts_log = []
            self.joint_measured_efforts_log = []
            self.joint_velocities_log = []

        """ 실시간 플로팅 시스템 초기화 """
        if (JOINT_PLOT_QT):
            self.plotter = RealTimePlotter()
            self.setup_data_signal()
        
        """ Contact Data 로깅을 위한 Contact Initialization """
        if (_LEFT_FOOT_COP):
            self.l1_contact_sensor = RigidContactView(
                prim_paths_expr="/World/envs/env_0/Robot/L_point[1-4]",
                name="Robot",
                filter_paths_expr=["/World/ground"]
            )
            self.l2_contact_sensor = RigidContactView(
                prim_paths_expr="/World/envs/env_0/Robot/l_hill",
                name="Robot",
                filter_paths_expr=["/World/ground"]
            )
            self.l1_contact_sensor.initialize()
            self.l2_contact_sensor.initialize()
            self.lf_cop_log = []  
        if (_RIGHT_FOOT_COP):

            self.r1_contact_sensor = RigidContactView(
                prim_paths_expr="/World/envs/env_0/Robot/R_point[1-4]",
                name="Robot",
                filter_paths_expr=["/World/ground"]
            )
            self.r2_contact_sensor = RigidContactView(
                prim_paths_expr="/World/envs/env_0/Robot/r_hill",
                name="Robot",
                filter_paths_expr=["/World/ground"]
            )
            self.r1_contact_sensor.initialize()
            self.r2_contact_sensor.initialize()
            self.rf_cop_log = []   
        if (COP_PLOT_QT):
                self.foot_cop_plotter = FootCOPPlotter()

        """ Robot Body Initialization """
        if (_ROBOT_BODY):
            self.robot_body = RobotView(
                prim_paths_expr="/World/envs/env_0/Robot",
                name="Robot"
            )
            self.robot_body.initialize()

            self.left_leg = RigidPrimView(
                prim_paths_expr="/World/envs/env_0/Robot/l.*",
                name="Robot"
            )
            self.left_leg.initialize()

            self.right_leg = RigidPrimView(
                prim_paths_expr="/World/envs/env_0/Robot/r.*",
                name="Robot"
            )
            self.right_leg.initialize()

            self.robot_body_data_log = []
            self.left_link_data_log = []
            self.right_link_data_log = []     
        if (_ROBOT_BODY_DATA_PLOT_QT):
            self.robot_body_data_plotter = RobotBodyDataPlotter()

    def setup_data_signal(self):
        """Qt 신호-슬롯 연결 설정"""
        self.plotter.data_updated.connect(self.plotter.update_plots)

    def reset(self, env_ids: Sequence[int]):
        pass

    def compute(
            self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # compute errors
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel

        # calculate the desired joint torques
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts

        # transfer actuator space 
        self._actuator_velo = torch.matmul(self.jacob_inv, joint_vel.T).T
        self._actuator_effort = torch.matmul(self.jacob_T, self.computed_effort.T).T
        self._cliped_actu_effort = self._clip_effort_actuator_space(self._actuator_effort)

        control_action.joint_efforts = torch.matmul(self.jacob_inv_T, self._cliped_actu_effort.T).T


        # apply friction model on the torque
        control_action.joint_efforts = control_action.joint_efforts - (self.friction_coulomb * torch.tanh(
            joint_vel / self.activation_vel) + self.friction_viscous * joint_vel)

        self.applied_effort = control_action.joint_efforts
        control_action.joint_positions = None
        control_action.joint_velocities = None

        """Robot Body Data Logging"""
        if (_ROBOT_BODY):
            robot_body_velocities = self.robot_body.get_linear_velocities().detach().cpu().numpy()
            robot_body_angular_velocities = self.robot_body.get_angular_velocities().detach().cpu().numpy()
            left_link_velocities = self.left_leg.get_velocities().detach().cpu().numpy()
            right_link_velocities = self.right_leg.get_velocities().detach().cpu().numpy()

            robot_body_world_poses, robot_body_orientations = self.robot_body.get_world_poses()
            robot_body_world_poses = robot_body_world_poses.detach().cpu().numpy()
            robot_body_orientations = robot_body_orientations.detach().cpu().numpy()

            left_link_world_poses, left_link_orientations = self.left_leg.get_world_poses()
            left_link_world_poses = left_link_world_poses.detach().cpu().numpy()
            left_link_orientations = left_link_orientations.detach().cpu().numpy()

            right_link_world_poses, right_link_orientations = self.right_leg.get_world_poses()
            right_link_world_poses = right_link_world_poses.detach().cpu().numpy()
            right_link_orientations = right_link_orientations.detach().cpu().numpy()
            
            # 오일러 각 변환
            robot_body_euler_angles = np.array([
                quat_to_euler_angles(q, degrees=False, extrinsic=True) for q in robot_body_orientations
            ])
            left_link_euler_angles = np.array([
                quat_to_euler_angles(q, degrees=False, extrinsic=True) for q in left_link_orientations
            ])
            right_link_euler_angles = np.array([
                quat_to_euler_angles(q, degrees=False, extrinsic=True) for q in right_link_orientations
            ])
            
            robot_body_euler_angles = rad2deg(robot_body_euler_angles)
            left_link_euler_angles = rad2deg(left_link_euler_angles)
            right_link_euler_angles = rad2deg(right_link_euler_angles)
    
            self.robot_body_data_log.append({
                "pos_x": robot_body_world_poses[0][0].item(),
                "pos_y": robot_body_world_poses[0][1].item(),
                "pos_z": robot_body_world_poses[0][2].item(),
                "ori_roll": robot_body_euler_angles[0][0],
                "ori_pitch": robot_body_euler_angles[0][1],
                "ori_yaw": robot_body_euler_angles[0][2],
                "linear_vel_x": robot_body_velocities[0][0].item(),
                "linear_vel_y": robot_body_velocities[0][1].item(),
                "linear_vel_z": robot_body_velocities[0][2].item(),
                "angular_vel_x": robot_body_angular_velocities[0][0].item(),
                "angular_vel_y": robot_body_angular_velocities[0][1].item(),
                "angular_vel_z": robot_body_angular_velocities[0][2].item(),
            })

            # 링크별 데이터 분리 저장
            left_link_names = ["l_hy", "l_hr", "l_hp", "l_kn", "l_ak", "l_ft", "l_hill", "l_to"]
            for idx, name in enumerate(left_link_names):
                if not hasattr(self, f'left_{name}_log'):
                    setattr(self, f'left_{name}_log', [])
                
                getattr(self, f'left_{name}_log').append({
                    "pos_x": left_link_world_poses[idx][0].item(),
                    "pos_y": left_link_world_poses[idx][1].item(),
                    "pos_z": left_link_world_poses[idx][2].item(),
                    "ori_roll": left_link_euler_angles[idx][0],
                    "ori_pitch": left_link_euler_angles[idx][1],
                    "ori_yaw": left_link_euler_angles[idx][2],
                    "linear_vel_x": left_link_velocities[idx][0].item(),
                    "linear_vel_y": left_link_velocities[idx][1].item(),
                    "linear_vel_z": left_link_velocities[idx][2].item(),
                    "angular_vel_x": left_link_velocities[idx][3].item(),
                    "angular_vel_y": left_link_velocities[idx][4].item(),
                    "angular_vel_z": left_link_velocities[idx][5].item(),   
                })

            right_link_names = ["r_hy", "r_hr", "r_hp", "r_kn", "r_ak", "r_ft", "r_hill", "r_to"]
            for idx, name in enumerate(right_link_names):
                if not hasattr(self, f'right_{name}_log'):
                    setattr(self, f'right_{name}_log', [])
                
                getattr(self, f'right_{name}_log').append({
                    "pos_x": right_link_world_poses[idx][0].item(),
                    "pos_y": right_link_world_poses[idx][1].item(),
                    "pos_z": right_link_world_poses[idx][2].item(),
                    "ori_roll": right_link_euler_angles[idx][0],
                    "ori_pitch": right_link_euler_angles[idx][1],
                    "ori_yaw": right_link_euler_angles[idx][2],
                    "linear_vel_x": right_link_velocities[idx][0].item(),
                    "linear_vel_y": right_link_velocities[idx][1].item(),
                    "linear_vel_z": right_link_velocities[idx][2].item(),
                    "angular_vel_x": right_link_velocities[idx][3].item(),
                    "angular_vel_y": right_link_velocities[idx][4].item(),
                    "angular_vel_z": right_link_velocities[idx][5].item(),
                })

                        # total_lf_cop_log = len(self.lf_cop_log)
            total_robot_body_data_log = len(self.robot_body_data_log)


            if total_robot_body_data_log % 1 == 0:
                progress_percent = (total_robot_body_data_log / BODY_DATA_LOG_MAX_COUNT) * 100
                print(f"수집 진행률: {total_robot_body_data_log}/{BODY_DATA_LOG_MAX_COUNT} ({progress_percent:.1f}%)          ", end='\r', flush=True)   
            if total_robot_body_data_log >= BODY_DATA_LOG_MAX_COUNT:
                self.save_robot_body_data_logs()


            # 만약 실시간 플롯을 사용한다면, RobotBodyDataPlotter 인스턴스를 생성한 후 업데이트합니다.
            if (_ROBOT_BODY_DATA_PLOT_QT):
                self.robot_body_data_plotter.update_plots(
                    self.step_counter,  # 현재 시간 전달
                    self.robot_body_data_log,
                    {name: getattr(self, f'left_{name}_log') for name in ["l_hy", "l_hr", "l_hp", "l_kn", "l_ak", "l_ft", "l_hill", "l_to"]},
                    {name: getattr(self, f'right_{name}_log') for name in ["r_hy", "r_hr", "r_hp", "r_kn", "r_ak", "r_ft", "r_hill", "r_to"]}
                )
                self.step_counter += 1
        
        """Robot Joint Data Logging"""
        if (_ALL_JOINT_LOG):

            joint_positions_tensor = self.robot.get_joint_positions()
            joint_applied_efforts_tensor = self.robot.get_applied_joint_efforts()
            joint_measured_efforts_tensor = self.robot.get_measured_joint_efforts()
            joint_velocities_tensor = self.robot.get_joint_velocities()

            joint_positions_np = joint_positions_tensor.detach().cpu().numpy()
            joint_applied_efforts_np = joint_applied_efforts_tensor.detach().cpu().numpy()
            joint_measured_efforts_np = joint_measured_efforts_tensor.detach().cpu().numpy()
            joint_velocities_np = joint_velocities_tensor.detach().cpu().numpy()

            row = joint_positions_np[0].tolist()
            row1 = joint_applied_efforts_np[0].tolist()
            row2 = joint_measured_efforts_np[0].tolist()
            row3 = joint_velocities_np[0].tolist()

            self.joint_positions_log.append(row)
            self.joint_applied_efforts_log.append(row1)
            self.joint_measured_efforts_log.append(row2)
            self.joint_velocities_log.append(row3)

            total_rows = len(self.joint_positions_log)
            total_rows_applied_efforts = len(self.joint_applied_efforts_log)
            total_rows_measured_efforts = len(self.joint_measured_efforts_log)
            total_rows_velocities = len(self.joint_velocities_log)

        """ 통합 데이터 수집 로직 (Joint Data) """
        if (_ALL_JOINT_LOG):
            import csv, os
            from datetime import datetime
            import sys

            # 로그 저장시 공통 output_dir 사용
            common_output_dir = "combined_logs"

            # 로깅 설정 리스트를 수정 (output_dir을 모두 'combined_logs'로)
            log_configs = [
                {
                    'log_name': 'positions',
                    'log_list': self.joint_positions_log,
                    'row_count': len(self.joint_positions_log),
                    'output_dir': common_output_dir
                },
                {
                    'log_name': 'applied_efforts',
                    'log_list': self.joint_applied_efforts_log,
                    'row_count': len(self.joint_applied_efforts_log),
                    'output_dir': common_output_dir
                },
                {
                    'log_name': 'measured_efforts',
                    'log_list': self.joint_measured_efforts_log,
                    'row_count': len(self.joint_measured_efforts_log),
                    'output_dir': common_output_dir
                },
                {
                    'log_name': 'velocities',
                    'log_list': self.joint_velocities_log,
                    'row_count': len(self.joint_velocities_log),
                    'output_dir': common_output_dir
                }
            ]

            for config in log_configs:
                current_count = config['row_count']
                if current_count % 1 == 0:
                    progress = (current_count / _LOG_THRESHOLD) * 100
                    print(f"{config['log_name']} 수집 진행률: {current_count}/{_LOG_THRESHOLD} ({progress:.1f}%)", end='\r', flush=True)
                
                if current_count >= _LOG_THRESHOLD:
                    os.makedirs(common_output_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_filename = f"joint_{config['log_name']}_{timestamp}.csv"
                    csv_path = os.path.join(common_output_dir, csv_filename)
                    with open(csv_path, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(self.robot.dof_names)
                        writer.writerows(config['log_list'])
                    
                    config['log_list'].clear()
                    print(f"\n{config['log_name']} 데이터 수집 완료. 파일 저장: {csv_path}")

        """ FOOT COP """
        if (_LEFT_FOOT_COP):
            left_toe_contact = self.l1_contact_sensor.get_net_contact_forces(dt=0.005)
            left_heel_contact = self.l2_contact_sensor.get_net_contact_forces(dt=0.005)

            # 모든 방향(x, y, z)의 접촉력 추출
            # toe1, toe2, toe3, toe4의 x, y, z 방향 힘 추출
            F_toe1_x = left_toe_contact[0, 0].item()
            F_toe1_y = left_toe_contact[0, 1].item()
            F_toe1_z = left_toe_contact[0, 2].item() 

            F_toe2_x = left_toe_contact[1, 0].item()
            F_toe2_y = left_toe_contact[1, 1].item()
            F_toe2_z = left_toe_contact[1, 2].item()

            F_toe3_x = left_toe_contact[2, 0].item()
            F_toe3_y = left_toe_contact[2, 1].item()
            F_toe3_z = left_toe_contact[2, 2].item()

            F_toe4_x = left_toe_contact[3, 0].item()
            F_toe4_y = left_toe_contact[3, 1].item()
            F_toe4_z = left_toe_contact[3, 2].item()

            F_heel_x = left_heel_contact[0, 0].item()
            F_heel_y = left_heel_contact[0, 1].item()
            F_heel_z = left_heel_contact[0, 2].item()

            # 발바닥 각 접촉점의 좌표 (mm 단위, Toe Pad의 중점을 원점으로)
            pos_toe1 = (-37.43, 33)
            pos_toe2 = (40.0, 32.69)
            pos_toe3 = (40.0, -32.69)
            pos_toe4 = (-37.43, -33)
            pos_heel = (-192.35, 0)    

            # 각 접촉점의 힘과 좌표를 리스트로 정리
            forces = [F_toe1_z, F_toe2_z, F_toe3_z, F_toe4_z, F_heel_z]

            # 전체 접촉력의 합 계산 (분모)
            total_force = sum(forces)

            if total_force != 0:
                # x 좌표에 대한 가중합과 y 좌표에 대한 가중합 계산
                weighted_sum_x = (
                    F_toe1_z * pos_toe1[0] + 
                    F_toe2_z * pos_toe2[0] + 
                    F_toe3_z * pos_toe3[0] + 
                    F_toe4_z * pos_toe4[0] + 
                    F_heel_z * pos_heel[0]
                )
                weighted_sum_y = (
                    F_toe1_z * pos_toe1[1] + 
                    F_toe2_z * pos_toe2[1] + 
                    F_toe3_z * pos_toe3[1] + 
                    F_toe4_z * pos_toe4[1] + 
                    F_heel_z * pos_heel[1]
                )
                # Center of Pressure (COP) 계산
                LF_COP_x = weighted_sum_x / total_force
                LF_COP_y = weighted_sum_y / total_force
            else:
                LF_COP_x, LF_COP_y = 0, 0

            # COP 계산 결과와 모든 방향(x, y, z)의 힘 데이터를 로그에 저장
            self.lf_cop_log.append([
                LF_COP_x, LF_COP_y, 
                F_toe1_x, F_toe1_y, F_toe1_z, 
                F_toe2_x, F_toe2_y, F_toe2_z, 
                F_toe3_x, F_toe3_y, F_toe3_z, 
                F_toe4_x, F_toe4_y, F_toe4_z,
                F_heel_x, F_heel_y, F_heel_z, 
                total_force
            ])
            total_lf_cop_log = len(self.lf_cop_log)
            if total_lf_cop_log % 1 == 0:
                progress_percent = (total_lf_cop_log / _LOG_THRESHOLD) * 100
                print(f"수집 진행률: {total_lf_cop_log}/{_LOG_THRESHOLD} ({progress_percent:.1f}%)          ", end='\r', flush=True)    
            if total_lf_cop_log >= _LOG_THRESHOLD:
                import csv, os
                from datetime import datetime
                output_dir = "combined_logs"
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = os.path.join(output_dir, f"lf_cop_{timestamp}.csv")
                with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    # 헤더에 LF_COP_x, LF_COP_y 옆에 F_toe1, F_toe2, F_toe3, F_toe4, F_heel, total_force 추가
                    writer.writerow([
                        "LF_COP_x", "LF_COP_y", 
                        "F_toe1_x", "F_toe1_y", "F_toe1_z", 
                        "F_toe2_x", "F_toe2_y", "F_toe2_z", 
                        "F_toe3_x", "F_toe3_y", "F_toe3_z", 
                        "F_toe4_x", "F_toe4_y", "F_toe4_z",
                        "F_heel_x", "F_heel_y", "F_heel_z", 
                        "total_force"
                    ])
                    writer.writerows(self.lf_cop_log)
                self.lf_cop_log.clear()
                import sys
                print()
                print("모든 데이터를 수집하였습니다. 프로그램을 종료합니다.")

            if (COP_PLOT_QT):
                self.foot_cop_plotter.update_left_plot(LF_COP_x, LF_COP_y)
        
        if (_RIGHT_FOOT_COP):
            right_toe_contact = self.r1_contact_sensor.get_net_contact_forces(dt=0.005)
            right_heel_contact = self.r2_contact_sensor.get_net_contact_forces(dt=0.005)

            # 모든 방향(x, y, z)의 접촉력 추출
            # toe1, toe2, toe3, toe4의 x, y, z 방향 힘 추출
            F_toe1_x = right_toe_contact[0, 0].item()
            F_toe1_y = right_toe_contact[0, 1].item()
            F_toe1_z = right_toe_contact[0, 2].item()
            
            F_toe2_x = right_toe_contact[1, 0].item()
            F_toe2_y = right_toe_contact[1, 1].item()
            F_toe2_z = right_toe_contact[1, 2].item()
            
            F_toe3_x = right_toe_contact[2, 0].item()
            F_toe3_y = right_toe_contact[2, 1].item()
            F_toe3_z = right_toe_contact[2, 2].item()
            
            F_toe4_x = right_toe_contact[3, 0].item()
            F_toe4_y = right_toe_contact[3, 1].item()
            F_toe4_z = right_toe_contact[3, 2].item()
            
            # heel의 x, y, z 방향 힘 추출
            F_heel_x = right_heel_contact[0, 0].item()
            F_heel_y = right_heel_contact[0, 1].item()
            F_heel_z = right_heel_contact[0, 2].item()
            
            # 발바닥 각 접촉점의 좌표 (mm 단위, Toe Pad의 중점을 원점으로)
            pos_toe1 = (-37.43, 33)
            pos_toe2 = (40.0, -32.69)
            pos_toe3 = (40.0, 32.69)
            pos_toe4 = (-37.43, -33)
            pos_heel = (-192.35, 0)    
            
            # COP 계산에는 z방향 힘만 사용 (기존 방식 유지)
            forces = [F_toe1_z, F_toe2_z, F_toe3_z, F_toe4_z, F_heel_z]

            # 전체 접촉력의 합 계산 (분모)
            total_force = sum(forces)

            if total_force != 0:
                # x 좌표에 대한 가중합과 y 좌표에 대한 가중합 계산
                weighted_sum_x = (
                    F_toe1_z * pos_toe1[0] + 
                    F_toe2_z * pos_toe2[0] + 
                    F_toe3_z * pos_toe3[0] + 
                    F_toe4_z * pos_toe4[0] + 
                    F_heel_z * pos_heel[0]
                )

                weighted_sum_y = (  
                    F_toe1_z * pos_toe1[1] + 
                    F_toe2_z * pos_toe2[1] + 
                    F_toe3_z * pos_toe3[1] + 
                    F_toe4_z * pos_toe4[1] + 
                    F_heel_z * pos_heel[1]
                )
                # Center of Pressure (COP) 계산
                RF_COP_x = weighted_sum_x / total_force
                RF_COP_y = weighted_sum_y / total_force
            else:
                RF_COP_x, RF_COP_y = 0, 0

            # COP 계산 결과와 모든 방향(x, y, z)의 힘 데이터를 로그에 저장
            self.rf_cop_log.append([
                RF_COP_x, RF_COP_y, 
                F_toe1_x, F_toe1_y, F_toe1_z, 
                F_toe2_x, F_toe2_y, F_toe2_z, 
                F_toe3_x, F_toe3_y, F_toe3_z, 
                F_toe4_x, F_toe4_y, F_toe4_z,
                F_heel_x, F_heel_y, F_heel_z, 
                total_force
            ])
            
            total_rf_cop_log = len(self.rf_cop_log)
            if total_rf_cop_log % 1 == 0:
                progress_percent = (total_rf_cop_log / _LOG_THRESHOLD) * 100
                print(f"수집 진행률: {total_rf_cop_log}/{_LOG_THRESHOLD} ({progress_percent:.1f}%)          ", end='\r', flush=True)     
            if total_rf_cop_log >= _LOG_THRESHOLD:
                import csv, os
                from datetime import datetime
                output_dir = "combined_logs"
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = os.path.join(output_dir, f"rf_cop_{timestamp}.csv")
                with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    # 헤더에 모든 방향(x, y, z)의 접촉력 추가
                    writer.writerow([
                        "RF_COP_x", "RF_COP_y", 
                        "F_toe1_x", "F_toe1_y", "F_toe1_z", 
                        "F_toe2_x", "F_toe2_y", "F_toe2_z", 
                        "F_toe3_x", "F_toe3_y", "F_toe3_z", 
                        "F_toe4_x", "F_toe4_y", "F_toe4_z",
                        "F_heel_x", "F_heel_y", "F_heel_z", 
                        "total_force"
                    ])
                    writer.writerows(self.rf_cop_log)
                self.rf_cop_log.clear()
                import sys
                print()
                print("모든 데이터를 수집하였습니다. 프로그램을 종료합니다.")
                sys.exit(0)

            if (COP_PLOT_QT):
                self.foot_cop_plotter.update_right_plot(RF_COP_x, RF_COP_y)


        """ 데이터 버퍼 업데이트 (기존 append 방식 대체) """
        if (JOINT_PLOT_QT):
            timestamp = time.time()
            self.data_buffer['positions'].append((timestamp, row))
            self.data_buffer['applied'].append((timestamp, row1)) 
            self.data_buffer['measured'].append((timestamp, row2))
            self.data_buffer['velocities'].append((timestamp, row3))
            
            # 신호 발생 (0.05초 간격으로 자동 스로틀링)
            self.plotter.data_updated.emit(self.data_buffer)

        return control_action
        
    

    def _clip_effort_actuator_space(self, effort: torch.Tensor) -> torch.Tensor:
        # compute torque limits

        # 각각의 조건별 min / max 값을 미리 계산
        max_if_velo_positive = self.effort_limit * (1.0 - self._actuator_velo / self.velocity_limit)

        min_if_velo_negative = self.effort_limit * (-1.0 - self._actuator_velo / self.velocity_limit)

        # velo >= 0 이면 "양수에 대한 min/max" 값을, 그렇지 않으면 "음수에 대한 min/max" 값을 고른다.
        max_effort = torch.where(
            self._actuator_velo >= 0,
            max_if_velo_positive,
            self.effort_limit
        )
        min_effort = torch.where(
            self._actuator_velo >= 0,
            -self.effort_limit,
            min_if_velo_negative
        )

        return torch.clip(torch.clip(effort, min=min_effort, max=max_effort), -self.effort_limit, self.effort_limit)


    def save_robot_body_data_logs(self):
        """링크별 데이터를 개별 CSV/Excel 파일로 저장하되, 모든 로그를 하나의 폴더('combined_logs')에 저장"""
        import csv, os, sys
        from datetime import datetime
        import pandas as pd

        # 모든 로그를 저장할 통합 폴더 경로 지정
        output_dir = "combined_logs"
        os.makedirs(output_dir, exist_ok=True)

        # 로봇 본체 데이터 저장 (CSV)
        if self.robot_body_data_log:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(output_dir, f"robot_body_{timestamp_str}.csv")
            headers = ["pos_x", "pos_y", "pos_z", "ori_roll", "ori_pitch", "ori_yaw",
                    "linear_vel_x", "linear_vel_y", "linear_vel_z",
                    "angular_vel_x", "angular_vel_y", "angular_vel_z"]
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(self.robot_body_data_log)
            self.robot_body_data_log.clear()

        # 왼쪽 링크별 데이터 저장 (Excel, 하나의 파일 내 여러 시트)
        left_link_names = ["l_hy", "l_hr", "l_hp", "l_kn", "l_ak", "l_ft", "l_hill", "l_to"]
        left_logs = {}
        for name in left_link_names:
            log = getattr(self, f'left_{name}_log', [])
            if log:
                left_logs[name] = log
                setattr(self, f'left_{name}_log', [])
        if left_logs:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_path = os.path.join(output_dir, f"left_links_{timestamp_str}.xlsx")
            with pd.ExcelWriter(excel_path) as writer:
                for name, log in left_logs.items():
                    headers = ["pos_x", "pos_y", "pos_z", "ori_roll", "ori_pitch", "ori_yaw",
                            "linear_vel_x", "linear_vel_y", "linear_vel_z",
                            "angular_vel_x", "angular_vel_y", "angular_vel_z"]
                    df = pd.DataFrame(log, columns=headers)
                    df.to_excel(writer, sheet_name=name, index=False)

        # 오른쪽 링크별 데이터 저장 (Excel, 하나의 파일 내 여러 시트)
        right_link_names = ["r_hy", "r_hr", "r_hp", "r_kn", "r_ak", "r_ft", "r_hill", "r_to"]
        right_logs = {}
        for name in right_link_names:
            log = getattr(self, f'right_{name}_log', [])
            if log:
                right_logs[name] = log
                setattr(self, f'right_{name}_log', [])
        if right_logs:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_path = os.path.join(output_dir, f"right_links_{timestamp_str}.xlsx")
            with pd.ExcelWriter(excel_path) as writer:
                for name, log in right_logs.items():
                    headers = ["pos_x", "pos_y", "pos_z", "ori_roll", "ori_pitch", "ori_yaw",
                            "linear_vel_x", "linear_vel_y", "linear_vel_z",
                            "angular_vel_x", "angular_vel_y", "angular_vel_z"]
                    df = pd.DataFrame(log, columns=headers)
                    df.to_excel(writer, sheet_name=name, index=False)

if (_ROBOT_BODY_DATA_PLOT_QT):
    class RobotBodyDataPlotter(QtCore.QObject):
        def __init__(self):
            super().__init__()
            # 왼쪽 및 오른쪽 링크별: 각각 하나의 창을 생성합니다.
            self.link_names = [
                "l_hy", "l_hr", "l_hp", "l_kn", "l_ak", "l_ft", "l_hill", "l_to",
                "r_hy", "r_hr", "r_hp", "r_kn", "r_ak", "r_ft", "r_hill", "r_to"
            ]
            self.windows = {}  # 각 링크에 대한 창 및 플롯 정보를 저장하는 dict
            for link in self.link_names:
                win = pg.GraphicsLayoutWidget()
                win.setWindowTitle(f"Robot Body Data - {link.upper()}")

                # 하나의 플롯에 두 개의 곡선을 생성 (위치: 녹색, 속도: 빨강)
                plot = win.addPlot(title=f"{link.upper()} Data")
                plot.setLabel('left', '값')
                plot.setLabel('bottom', '시간 (step)')
                plot.showGrid(x=True, y=True)

                # pos_x, pos_y, pos_z 별로 각각 곡선을 생성합니다.
                plot.addLegend(offset=(10, 10))
                pos_curve_x = plot.plot([], [], pen=pg.mkPen(color=(0, 255, 0), width=2), name="Pos_X")
                pos_curve_y = plot.plot([], [], pen=pg.mkPen(color=(0, 128, 255), width=2), name="Pos_Y")
                pos_curve_z = plot.plot([], [], pen=pg.mkPen(color=(255, 0, 0), width=2), name="Pos_Z")
                # 속도용 곡선은 필요한 경우 사용합니다.
                vel_curve = plot.plot([], [], pen=pg.mkPen(color=(255, 0, 0), width=2), name="Velocity")
               
                self.windows[link] = {
                    "win": win,
                    "plot": plot,
                    "pos_curve_x": pos_curve_x,
                    "pos_curve_y": pos_curve_y,
                    "pos_curve_z": pos_curve_z,
                    "vel_curve": vel_curve,
                    "time": [],
                    "pos_x_data": [],
                    "pos_y_data": [],
                    "pos_z_data": [],
                    "vel_data": []
                }
                win.show()

        def update_plots(self, current_time, robot_body_log, left_links_log, right_links_log):
            """
            각 링크별 최신 데이터를 반영하여 해당 창의 그래프를 업데이트합니다.
            left_links_log, right_links_log은 {링크 이름: 로그 리스트} 형태로 제공되며,
            각 로그 항목은 최소한 'pos_x'와 'pos_y', 'pos_z' 키를 포함해야 합니다.
            """
            # 왼쪽 링크 업데이트
            for link, log_list in left_links_log.items():
                if log_list:
                    latest_data = log_list[-1]
                    pos_x = latest_data.get('pos_x', 0)
                    pos_y = latest_data.get('pos_y', 0)
                    pos_z = latest_data.get('pos_z', 0)
                    self._update_link_plot(link, current_time, pos_x, pos_y, pos_z)
            # 오른쪽 링크 업데이트
            for link, log_list in right_links_log.items():
                if log_list:
                    latest_data = log_list[-1]
                    pos_x = latest_data.get('pos_x', 0)
                    pos_y = latest_data.get('pos_y', 0)
                    pos_z = latest_data.get('pos_z', 0)
                    self._update_link_plot(link, current_time, pos_x, pos_y, pos_z)

        def _update_link_plot(self, link_name, time, pos_x, pos_y, pos_z):
            """
            지정된 링크의 창에 데이터를 추가하여, 각 좌표에 대해 별도의 곡선으로 업데이트합니다.
            여기서 time 배열과 각각의 pos_data 배열의 길이는 동일해야 하므로, 각 step마다 한 개의 값씩 추가합니다.
            """
            max_points = 300
            if link_name in self.windows:
                data = self.windows[link_name]
                data["time"].append(time)
                data["pos_x_data"].append(pos_x)
                data["pos_y_data"].append(pos_y)
                data["pos_z_data"].append(pos_z)

                # 최대 데이터 포인트를 넘으면 최근 max_points만 남깁니다.
                if len(data["time"]) > max_points:
                    data["time"] = data["time"][-max_points:]
                    data["pos_x_data"] = data["pos_x_data"][-max_points:]
                    data["pos_y_data"] = data["pos_y_data"][-max_points:]
                    data["pos_z_data"] = data["pos_z_data"][-max_points:]
                
                data["pos_curve_x"].setData(data["time"], data["pos_x_data"])
                data["pos_curve_y"].setData(data["time"], data["pos_y_data"])
                data["pos_curve_z"].setData(data["time"], data["pos_z_data"])

if (COP_PLOT_QT):
    class FootCOPPlotter(QtCore.QObject):
        """
        LF_COP 및 RF_COP 실시간 2D Plotter 클래스 (Subplots)

        - 왼쪽 서브플롯: Left Foot COP (X축: LF_COP_y, Y축: LF_COP_x)
        - 오른쪽 서브플롯: Right Foot COP (X축: RF_COP_y, Y축: RF_COP_x)
        """
        def __init__(self):
            super().__init__()
            # GraphicsLayoutWidget을 이용해 하나의 창에 두 개의 서브플롯 생성
            self.win = pg.GraphicsLayoutWidget()
            self.win.setWindowTitle("LF & RF COP 실시간 Plot")
            
            # 왼쪽발 플롯 생성
            self.left_plot = self.win.addPlot(title="Left Foot COP", row=0, col=0)
            self.left_plot.setLabel('left', "LF_COP_x (세로)")
            self.left_plot.setLabel('bottom', "LF_COP_y (가로)")
            self.left_plot.getViewBox().invertX()  # X축 반전 (오른쪽이 + → 왼쪽이 +)
            self.left_plot.setXRange(-45, 45, padding=0)
            self.left_plot.setYRange(-200, 45, padding=0)
            self.left_curve = self.left_plot.plot([], [], pen=pg.mkPen(color=(0, 255, 0), width=2))
            self.left_x_data = []
            self.left_y_data = []
            
            # 오른쪽발 플롯 생성
            self.right_plot = self.win.addPlot(title="Right Foot COP", row=0, col=1)
            self.right_plot.setLabel('left', "RF_COP_x (세로)")
            self.right_plot.setLabel('bottom', "RF_COP_y (가로)")
            self.right_plot.getViewBox().invertX()  # X축 반전
            self.right_plot.setXRange(-45, 45, padding=0)
            self.right_plot.setYRange(-200, 45, padding=0)
            self.right_curve = self.right_plot.plot([], [], pen=pg.mkPen(color=(255, 0, 0), width=2))
            self.right_x_data = []
            self.right_y_data = []
            
            self.win.show()
        
        def update_left_plot(self, LF_COP_x: float, LF_COP_y: float):
            """
            왼쪽발 COP 데이터 업데이트
            - X축: LF_COP_y (가로)
            - Y축: LF_COP_x (세로)
            (LF_COP_x, LF_COP_y)가 (0, 0)인 경우, 해당 플롯 데이터를 초기화함
            """
            if LF_COP_x == 0 and LF_COP_y == 0:
                self.left_x_data = []
                self.left_y_data = []
                self.left_curve.setData(self.left_x_data, self.left_y_data)
            else:
                self.left_x_data.append(LF_COP_y)
                self.left_y_data.append(LF_COP_x)
                self.left_curve.setData(self.left_x_data, self.left_y_data)
        
        def update_right_plot(self, RF_COP_x: float, RF_COP_y: float):
            """
            오른쪽발 COP 데이터 업데이트
            - X축: RF_COP_y (가로)
            - Y축: RF_COP_x (세로)
            (RF_COP_x, RF_COP_y)가 (0, 0)인 경우, 해당 플롯 데이터를 초기화함
            """
            if RF_COP_x == 0 and RF_COP_y == 0:
                self.right_x_data = []
                self.right_y_data = []
                self.right_curve.setData(self.right_x_data, self.right_y_data)
            else:
                self.right_x_data.append(RF_COP_y)
                self.right_y_data.append(RF_COP_x)
                self.right_curve.setData(self.right_x_data, self.right_y_data)

if JOINT_PLOT_QT:
    class RealTimePlotter(QtCore.QObject):
        data_updated = QtCore.pyqtSignal(dict)

        def __init__(self, legs_to_plot="left"):
            """
            legs_to_plot: 'both' (기본값), 'left', 'right'
            """
            super().__init__()
            self.legs_to_plot = legs_to_plot  # 플롯할 다리 설정
            self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            self.win = pg.GraphicsLayoutWidget()
            self.plots = self.create_plots()  # 플롯과 라인 객체를 미리 생성
            self.win.show()

        def create_plots(self):
            """3개의 서브플롯과 곡선(line)을 미리 생성하여 성능 최적화.
            각 곡선의 이름은 사전에 정의된 관절 이름을 사용합니다.
            순서: Positions, Velocities, Measured Efforts
            """
            # 전체 관절 이름 리스트
            joint_names_all = ['L_HY', 'R_HY', 'L_HR', 'R_HR', 'L_HP', 'R_HP',
                            'L_KN', 'R_KN', 'L_AK', 'R_AK', 'left_leg', 'R_FT',
                            'L_TO', 'R_TO']
            # 선택한 다리에 해당하는 인덱스 결정
            if self.legs_to_plot == "left":
                self.joint_indices = [i for i, name in enumerate(joint_names_all) if name.startswith("L_")]
            elif self.legs_to_plot == "right":
                self.joint_indices = [i for i, name in enumerate(joint_names_all) if name.startswith("R_")]
            else:
                self.joint_indices = list(range(len(joint_names_all)))
            
            # 필터링된 관절 이름 리스트
            joint_names = [joint_names_all[i] for i in self.joint_indices]
            
            plots = []
            self.curves = []  # 각 서브플롯에 대한 곡선 리스트 저장
            titles = ['Positions', 'Velocities', 'Measured Efforts']  # 플롯 순서 지정
            for i, title in enumerate(titles):
                plot = self.win.addPlot(title=title, row=i, col=0)
                plot.setLabel('left', 'Value')
                plot.setLabel('bottom', 'Step')
                plot.addLegend()
                plot.showGrid(x=True, y=True)
                plots.append(plot)
                
                curves_for_plot = []
                for joint_idx, joint_name in enumerate(joint_names):
                    curve = plot.plot([], [], pen=pg.intColor(joint_idx), name=joint_name)
                    curves_for_plot.append(curve)
                self.curves.append(curves_for_plot)
            return plots

        def update_plots(self, data):
            """0.05초마다 호출되는 업데이트 함수.
            x축은 로그가 쌓이는 단계(step)를 사용하며, 
            업데이트할 데이터는 'positions', 'velocities', 'measured' 순서로 처리합니다.
            선택된 다리의 인덱스(self.joint_indices)를 사용하여 데이터를 필터링합니다.
            """
            keys = ['positions', 'velocities', 'measured']
            for idx, key in enumerate(keys):
                if data[key]:
                    # x축은 로그가 쌓인 순서를 단계(step)로 사용 (0부터 N-1까지)
                    steps = np.arange(len(data[key]))
                    full_values = np.array([d[1] for d in data[key]])  # shape: (N, 14)
                    # 선택한 다리의 데이터만 추출 (예: left면 [0,2,4,...])
                    filtered_values = full_values[:, self.joint_indices]
                    for joint_idx, curve in enumerate(self.curves[idx]):
                        curve.setData(steps, filtered_values[:, joint_idx])