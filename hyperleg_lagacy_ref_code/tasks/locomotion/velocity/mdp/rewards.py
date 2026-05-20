# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.core.utils.torch.rotations import quat_apply
from omni.isaac.lab.assets import Articulation

if TYPE_CHECKING:
    from omni.isaac.lab.envs import RLTaskEnv

# def feet_air_time(env: RLTaskEnv, command_name: str, 
#                   sensor_cfg: SceneEntityCfg, 
#                   threshold_min: float,
#                   threshold_max: float,
#                   alternating_bonus: float = 2.0) -> torch.Tensor:
#     """Reward long steps taken by the feet using L2-kernel.

#     This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
#     that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
#     the time for which the feet are in the air.

#     If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
#     """
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     # negative reward for small steps
#     air_time = (last_air_time - threshold_min) * first_contact
#     # no reward for large steps
#     air_time = torch.clamp(air_time, max=threshold_max - threshold_min)
#     reward = torch.sum(air_time, dim=1)
#     # no reward for zero command
#     reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
#     return reward

def feet_air_time(env: RLTaskEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold_min: float,
                  threshold_max: float, debug: bool = False) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # negative reward for small steps
    air_time = (last_air_time - threshold_min) * first_contact
    # no reward for large steps
    air_time = torch.clamp(air_time, max=threshold_max - threshold_min)
    reward = torch.sum(air_time, dim=1)
    # no reward for zero command
    reward_mask = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    reward = reward * reward_mask
    
    # 디버그 출력 (first_contact가 활성화되었을 때만)
    if debug:
        nonzero_indices = torch.nonzero(first_contact[0])
        if len(nonzero_indices) > 0:  # first_contact가 활성화된 경우에만 출력
            print("\n----- Feet Air Time Reward Debug -----")
            for idx in nonzero_indices:
                body_idx = idx.item()
                raw_air_time = last_air_time[0, body_idx].item()
                clamped_air_time = air_time[0, body_idx].item()
                
                print(f"Body {body_idx}: Air Time = {raw_air_time:.3f}s")
                print(f"  - Threshold Applied: {raw_air_time - threshold_min:.3f}s")
                print(f"  - Clamped Air Time: {clamped_air_time:.3f}")
            
            print(f"Total Air Time Reward: {reward[0].item():.3f}")
            if not reward_mask[0]:
                print("  (Zero command detected - reward scaled to zero)")
            print("--------------------------------------")
    
    return reward

def feet_slide(
        env, 
        sensor_cfg: SceneEntityCfg, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    panalty = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return panalty

def foot_contact(
        env: RLTaskEnv, 
        left_sensor_cfg: SceneEntityCfg,
        right_sensor_cfg: SceneEntityCfg,
        contact_threshold: float = 0.0001,
        penalty_scale: float = -0.1,
        debug: bool = False
    ) -> torch.Tensor:

    # 왼발 접촉 처리
    left_contact_sensor: ContactSensor = env.scene.sensors[left_sensor_cfg.name]
    left_contact_time = left_contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids]
    left_mask = left_contact_time > contact_threshold
    left_contact_count = torch.sum(left_mask, dim=1)
 
    # 오른발 접촉 처리
    right_contact_sensor: ContactSensor = env.scene.sensors[right_sensor_cfg.name]
    right_contact_time = right_contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids]
    right_mask = right_contact_time > contact_threshold
    right_contact_count = torch.sum(right_mask, dim=1)

    # 보상 조건: 3점 이상 접촉
    left_reward_condition = (left_contact_count >= 3)
    right_reward_condition = (right_contact_count >= 3)

    # 발 접촉 상태 변수 저장 (번갈아 사용 추적용)
    if not hasattr(env, "last_active_foot"):
        env.last_active_foot = torch.zeros_like(left_reward_condition, dtype=torch.int32)  # 0: 없음, 1: 왼발, 2: 오른발
        env.prev_left_contact = torch.zeros_like(left_reward_condition)
        env.prev_right_contact = torch.zeros_like(right_reward_condition)

    # 새로 접촉한 발 감지
    left_new_contact = left_reward_condition & ~env.prev_left_contact
    right_new_contact = right_reward_condition & ~env.prev_right_contact

     # 현재 활성화된 발 확인
    current_active_foot = torch.where(
        left_new_contact, 
        torch.ones_like(left_reward_condition, dtype=torch.int32),
        torch.where(
            right_new_contact,
            torch.ones_like(right_reward_condition, dtype=torch.int32) * 2,
            torch.zeros_like(left_reward_condition, dtype=torch.int32)
        )
    )
    
    # 번갈아 사용 확인: 새로 접촉된 발이 있고, 이전 접촉 발과 다른 경우
    is_alternating = (current_active_foot > 0) & (env.last_active_foot > 0) & (current_active_foot != env.last_active_foot)
    
    # 이전 상태 업데이트 (현재 발 활성화가 있을 때만)
    env.last_active_foot = torch.where(
        current_active_foot > 0,
        current_active_foot,
        env.last_active_foot
    )

    left_reward = torch.where(left_reward_condition, 
                             torch.ones_like(left_contact_count),  # 3점 이상 접촉시 보상 1
                             torch.zeros_like(left_contact_count))
    
    right_reward = torch.where(right_reward_condition,
                              torch.ones_like(right_contact_count),
                              torch.zeros_like(right_contact_count))
    
    # 왼발 예외 조건 처리
    left_exception1 = left_mask[:, 0] & left_mask[:, 1] & left_mask[:, 2]
    left_exception2 = left_mask[:, 0] & left_mask[:, 3] & left_mask[:, 4]
    left_exception = left_exception1 | left_exception2
    left_reward = torch.where(left_exception, torch.ones_like(left_reward) * penalty_scale, left_reward)
    
    # 오른발 예외 조건 처리
    right_exception1 = right_mask[:, 0] & right_mask[:, 1] & right_mask[:, 3]
    right_exception2 = right_mask[:, 0] & right_mask[:, 2] & right_mask[:, 4]
    right_exception = right_exception1 | right_exception2
    right_reward = torch.where(right_exception, torch.ones_like(right_reward) * penalty_scale, right_reward)

    # 최종 보상: 발이 번갈아 사용될 때만 보상을 적용
    left_reward = torch.where(is_alternating, left_reward, torch.zeros_like(left_reward))
    right_reward = torch.where(is_alternating, right_reward, torch.zeros_like(right_reward))

    # 최종 보상 (양발 보상 합산)
    total_reward = left_reward + right_reward

    # 이전 접촉 상태 저장
    env.prev_left_contact = left_reward_condition.clone()
    env.prev_right_contact = right_reward_condition.clone()
    
    if debug & (env.common_step_counter % 24 == 0):
        
        print(f"\n-- Left foot --")
        print(f"Contacts: {left_contact_time[0]}")
        print(f"Contact count: {left_contact_count[0]}")
        print(f"Reward: {left_reward[0]:.3f}")
        print(f"\n-- Right foot --")
        print(f"Contacts: {right_contact_time[0]}")
        print(f"Contact count: {right_contact_count[0]}")
        print(f"Reward: {right_reward[0]:.3f}")
        print(f"\n-- Total reward: {total_reward[0]:.3f}")
    
    return total_reward

# def left_foot_contact(
#         env: RLTaskEnv, 
#         sensor_cfg: SceneEntityCfg,
#         contact_threshold: float = 0.0001,
#         reward_scale: float = 1.0,
#         debug: bool = False
#     ) -> torch.Tensor:
#     """왼발의 접촉 상태에 따른 보상 부여.
    
#     - 기본적으로 왼발의 5개 접촉 센서 중 3개 이상 접촉 시 reward_scale 만큼 보상.
#     - 단, 다음 예외 조건이 성립하면 -reward_scale의 패널티를 적용:
#         1. l_hill, L_point1, L_point2 (인덱스 0, 1, 2) 모두 접촉
#         2. l_hill, L_point3, L_point4 (인덱스 0, 3, 4) 모두 접촉
#     """
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

#     # 각 센서에 대해 접촉 여부 Boolean mask (shape: [batch, 5])
#     left_mask = contact_time > contact_threshold

#     # 접촉 센서 개수 계산
#     left_contact_count = torch.sum(left_mask, dim=1)

#     # 기본 보상: 3개 이상 접촉이면 reward_scale (그외 0)
#     reward = torch.where(left_contact_count >= 3,
#                          torch.ones_like(left_contact_count) * reward_scale,
#                          torch.zeros_like(left_contact_count))
    
#     # 예외조건 1: [0, 1, 2] 모두 접촉
#     exception1 = left_mask[:, 0] & left_mask[:, 1] & left_mask[:, 2]
#     # 예외조건 2: [0, 3, 4] 모두 접촉
#     exception2 = left_mask[:, 0] & left_mask[:, 3] & left_mask[:, 4]
    
#     # 예외 조건이 한쪽이라도 성립하면 패널티 적용 (음수 값)
#     exception_condition = exception1 | exception2
#     reward = torch.where(exception_condition, -torch.ones_like(reward) * 0.1, reward)
    
#     if debug:
#         print(f"\nLeft foot contacts: {contact_time[0]}")
#         print(f"Left contact count: {left_contact_count[0]}")
#         print(f"Reward: {reward[0]:.3f}")

#     return reward

# def right_foot_contact(
#         env: RLTaskEnv, 
#         sensor_cfg: SceneEntityCfg,
#         contact_threshold: float = 0.0001,
#         reward_scale: float = 1.0,
#         debug: bool = False
#     ) -> torch.Tensor:
#     """오른발의 접촉 상태에 따른 보상 부여.
    
#     - 기본적으로 오른발의 5개 접촉 센서 중 3개 이상 접촉 시 reward_scale 만큼 보상.
#     - 단, 다음 예외 조건이 성립하면 -reward_scale의 패널티를 적용:
#         3. r_hill, R_point1, R_point3 (인덱스 0, 1, 3) 모두 접촉
#         4. r_hill, R_point2, R_point4 (인덱스 0, 2, 4) 모두 접촉
#     """
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

#     # 각 센서에 대해 접촉 여부 Boolean mask (shape: [batch, 5])
#     right_mask = contact_time > contact_threshold

#     # 접촉 센서 개수 계산
#     right_contact_count = torch.sum(right_mask, dim=1)

#     # 기본 보상: 3개 이상 접촉이면 reward_scale (그외 0)
#     reward = torch.where(right_contact_count >= 3,
#                          torch.ones_like(right_contact_count) * reward_scale,
#                          torch.zeros_like(right_contact_count))
    
#     # 예외조건 1: r_hill, R_point1, R_point3 → 인덱스 [0, 1, 3] 모두 접촉
#     exception1 = right_mask[:, 0] & right_mask[:, 1] & right_mask[:, 3]
#     # 예외조건 2: r_hill, R_point2, R_point4 → 인덱스 [0, 2, 4] 모두 접촉
#     exception2 = right_mask[:, 0] & right_mask[:, 2] & right_mask[:, 4]
    
#     # 예외 조건 성립 시 패널티 적용
#     exception_condition = exception1 | exception2
#     reward = torch.where(exception_condition, -torch.ones_like(reward) * 0.1, reward)
    
#     if debug:
#         print(f"\nRight foot contacts: {contact_time[0]}")
#         print(f"Right contact count: {right_contact_count[0]}")
#         print(f"Reward: {reward[0]:.3f}")

#     return reward


JACOBIAN_T = torch.tensor([
    [0.04, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.04, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.04, 0.000, 0.000, 0.000, 0.000],
    [0.0, 0.0, 0.0, -0.0321, -0.0375,  0.0000, -0.0131],
    [0.0, 0.0, 0.0, 0.0, 0.0187, 0.0303, -0.0169],
    [0.0, 0.0, 0.0, 0.0, 0.0187, -0.0303, -0.0169],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0470]
], dtype=torch.float32).to("cuda:0")

JACOBIAN_INV = torch.tensor([
    [25.0, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 25.0, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 25.0, 0.000, 0.000, 0.000, 0.000],
    [0.0, 0.0, 0.0, -31.11,   0.00,   0.00,   0.00],
    [0.0, 0.0, 0.0, -31.11,  26.69,  16.51,   0.00],
    [0.0, 0.0, 0.0, -31.11,  26.69, -16.51,   0.00],
    [0.0, 0.0, 0.0, -31.11,  19.23,   0.00,  21.27]
], dtype=torch.float32).to("cuda:0")

coef_RI60 = 124.567474
coef_RI70 = 33.44203328
coef_U10 = 3.54148162

POWER_COEF = torch.tensor([
    [coef_U10, coef_U10, coef_U10, coef_RI70, coef_RI60, coef_RI60, coef_RI60]
], dtype=torch.float32).to("cuda:0")


def actuator_torques_l2(
        env: RLTaskEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 관절 토크 데이터 접근
    joint_torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # 왼쪽 다리 인덱스 (짝수): [0, 2, 4, 6, 8, 10, 12]
    # 오른쪽 다리 인덱스 (홀수): [1, 3, 5, 7, 9, 11, 13]
    joint_torques_left = joint_torques[:, ::2]    
    joint_torques_right = joint_torques[:, 1::2]
    joint_vel_left = joint_vel[:, ::2]
    joint_vel_right = joint_vel[:, 1::2]

    actuator_torques_left = torch.matmul(JACOBIAN_T, joint_torques_left.T).T
    actuator_torques_right = torch.matmul(JACOBIAN_T, joint_torques_right.T).T
    actuator_vel_left = torch.matmul(JACOBIAN_INV, joint_vel_left.T).T
    actuator_vel_right = torch.matmul(JACOBIAN_INV, joint_vel_right.T).T

    actuator_torques_left = torch.where(    
        actuator_torques_left < 0,
        actuator_torques_left * 0.9,
        actuator_torques_left / 0.8
    )

    actuator_torques_right = torch.where(
        actuator_torques_right < 0,
        actuator_torques_right * 0.9,
        actuator_torques_right / 0.8
    )

    # 토크 제곱 계산
    squared_torques_left = torch.square(actuator_torques_left)
    squared_torques_right = torch.square(actuator_torques_right)

    power_left = POWER_COEF * squared_torques_left
    power_right = POWER_COEF * squared_torques_right

    mech_work_left = actuator_torques_left * actuator_vel_left
    mech_work_left = torch.where(
        mech_work_left < 0, 
        mech_work_left * 0.9, 
        mech_work_left /0.9
    )

    mech_work_right = actuator_torques_right * actuator_vel_right
    mech_work_right = torch.where(
        mech_work_right < 0, 
        mech_work_right * 0.9, 
        mech_work_right /0.9
    )

    joule_heating_left = torch.sum(power_left, dim=1)
    joule_heating_right = torch.sum(power_right, dim=1)
    mech_work_left = torch.sum(mech_work_left, dim=1)
    mech_work_right = torch.sum(mech_work_right, dim=1)

    penalty = joule_heating_left + joule_heating_right + mech_work_left + mech_work_right

    return penalty