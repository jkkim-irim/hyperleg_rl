# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter

if TYPE_CHECKING:
    from omni.isaac.lab.envs import RLTaskEnv


def terrain_levels_vel(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

def modify_reward_weight(
        env: RLTaskEnv, 
        env_ids: Sequence[int], 
        term_name: str, 
        weight: float, 
        num_steps: int):
    """Curriculum that modifies a reward weight after some number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

def modify_push_force(
        env: RLTaskEnv, 
        env_ids: Sequence[int], 
        term_name: str, 
        max_velocity: Sequence[float], 
        interval: int, 
        starting_step: float = 0.0
        ):
    """Curriculum that modifies the maximum push (perturbation) velocity over some intervals. 

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        max_velocity: The maximum velocity of the push.
        interval: The number of steps after which the condition is checked again
        starting_step: The number of steps after which the curriculum is applied.
    """
    try:
        term_cfg = env.event_manager.get_term_cfg('push_robot')
    except:
        # print("No push_robot term found in the event manager")
        return 0.0
    curr_setting = term_cfg.params['velocity_range']['x'][1]
    if env.common_step_counter < starting_step:
        return curr_setting
    if env.common_step_counter % interval == 0:

        
        if torch.sum(env.termination_manager._term_dones["base_contact"]) < torch.sum(env.termination_manager._term_dones["time_out"]) * 2:
            # obtain term settings
            term_cfg = env.event_manager.get_term_cfg('push_robot')
            # update term settings
            curr_setting = term_cfg.params['velocity_range']['x'][1]
            curr_setting = np.clip(curr_setting * 1.5, 0.0, max_velocity[0])
            term_cfg.params['velocity_range']['x'] = (-curr_setting, curr_setting)
            curr_setting = term_cfg.params['velocity_range']['y'][1]
            curr_setting = np.clip(curr_setting * 1.5, 0.0, max_velocity[1])
            term_cfg.params['velocity_range']['y'] = (-curr_setting, curr_setting)
            env.event_manager.set_term_cfg('push_robot', term_cfg)
        

        if torch.sum(env.termination_manager._term_dones["base_contact"]) > torch.sum(env.termination_manager._term_dones["time_out"]) / 2:
            # obtain term settings
            term_cfg = env.event_manager.get_term_cfg('push_robot')
            # update term settings
            curr_setting = term_cfg.params['velocity_range']['x'][1]
            curr_setting = np.clip(curr_setting - 0.2, 0.0, max_velocity[0])
            term_cfg.params['velocity_range']['x'] = (-curr_setting, curr_setting)
            curr_setting = term_cfg.params['velocity_range']['y'][1]
            curr_setting = np.clip(curr_setting - 0.2, 0.0, max_velocity[1])
            term_cfg.params['velocity_range']['y'] = (-curr_setting, curr_setting)
            env.event_manager.set_term_cfg('push_robot', term_cfg)

    return curr_setting

def modify_command_velocity(
    env: RLTaskEnv, 
    env_ids: Sequence[int], 
    lin_vel_term_name: str, 
    max_lin_vel_x: Sequence[float], 
    max_lin_vel_y: Sequence[float], 
    interval: int, 
    starting_step: float = 0.0,
    lin_vel_threshold: float = 0.0,
    debug: bool = False
    ):
    """Curriculum that modifies the maximum command velocity over some intervals. 
    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        max_velocity: The maximum velocity. 
        interval: The number of steps after which the condition is checked again
        starting_step: The number of steps after which the curriculum is applied.
    """

    command_cfg = env.command_manager.get_term('base_velocity').cfg
    curr_lin_vel_x = command_cfg.ranges.lin_vel_x
    curr_lin_vel_y = command_cfg.ranges.lin_vel_y

    if env.common_step_counter < starting_step:
        return curr_lin_vel_x[1]
    
    if env.common_step_counter % interval == 0:
        rew_lin = env.reward_manager._episode_sums[lin_vel_term_name][env_ids]
        track_lin_vel_mean_reward = torch.mean(rew_lin) / env.max_episode_length_s
        if track_lin_vel_mean_reward > lin_vel_threshold:
            curr_lin_vel_x = (
                np.clip(curr_lin_vel_x[0] - 0.25, max_lin_vel_x[0], 0.0),
                np.clip(curr_lin_vel_x[1] + 0.5, 0.0, max_lin_vel_x[1])   # 최대속도 증가
            )
            curr_lin_vel_y = (
                np.clip(curr_lin_vel_y[0] - 0.25, max_lin_vel_y[0], 0.0),
                np.clip(curr_lin_vel_y[1] + 0.25, 0.0, max_lin_vel_y[1])   # 최대속도 증가
            )
            command_cfg.ranges.lin_vel_x = curr_lin_vel_x
            command_cfg.ranges.lin_vel_y = curr_lin_vel_y
        
        if debug:
            print(f"curr_lin_vel_x: {curr_lin_vel_x[0],curr_lin_vel_x[1]}, "
                  f"curr_lin_vel_y: {curr_lin_vel_y[0],curr_lin_vel_y[1]}")
        
    return curr_lin_vel_x[1]

def modify_weight_based_on_velocity(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    actuator_weight: float = -0.0001,
    lin_vel_z_l2_weight: float = -1.0
):
    command_cfg = env.command_manager.get_term('base_velocity').cfg
    curr_lin_vel_x = command_cfg.ranges.lin_vel_x
    if curr_lin_vel_x[1] >= 1.5:
        allowed_terms = {
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "actuator_torques_l2",
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "feet_air_time",
            "foot_contact",
            "feet_slide",
            "undesired_contacts",
        }
        # velocity_env_cfg.py 파일에 정의된 모든 보상 항목 목록
        all_terms = [
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "actuator_torques_l2",
            "action_rate_l2",
            "feet_air_time",
            "feet_slide",
            "foot_contact",
            "undesired_contacts",
            "joint_deviation_hip",
            "joint_deviation_knee",
            "joint_deviation_foot",
        ]
        for term_name in all_terms:
            term_cfg = env.reward_manager.get_term_cfg(term_name)
            if term_name in allowed_terms:
                if term_name == "actuator_torques_l2":
                    if term_cfg.weight != actuator_weight:
                        term_cfg.weight = actuator_weight
                        env.reward_manager.set_term_cfg(term_name, term_cfg)
                elif term_name == "lin_vel_z_l2":
                    if term_cfg.weight != lin_vel_z_l2_weight:
                        term_cfg.weight = lin_vel_z_l2_weight
                        env.reward_manager.set_term_cfg(term_name, term_cfg)
            else:
                if term_cfg.weight != 0.0:
                    term_cfg.weight = 0.0
                    env.reward_manager.set_term_cfg(term_name, term_cfg)
        return 
