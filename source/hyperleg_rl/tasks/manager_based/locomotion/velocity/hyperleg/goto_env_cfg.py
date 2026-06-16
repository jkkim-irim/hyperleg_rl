# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end goal-reaching locomotion env for HyperLeg ("goto").

A single policy directly controls the 14 leg joints (same action term as the
velocity-tracking walker) AND learns to walk to a commanded SE(2) pose goal —
i.e. walking and navigation are learned simultaneously from scratch. This avoids
the frozen-walker speed cap of the hierarchical ``Ttest`` task: the policy is free
to discover whatever (faster) gait gets it to the goal soonest.

It reuses ``HyperLegEnvCfg`` wholesale (scene, joint action, events, terminations,
sim, power/thermal penalties) and only swaps the task signal:
    * command:     base velocity  -> relative pose goal (UniformPose2dCommand)
    * observation: velocity_commands -> pose_command
    * reward:      velocity tracking -> position/heading tracking

Reward weights and hardware specs (hyperleg.py) are left as sensible defaults for
the user to tune. Trains on flat plane (the velocity-command terrain curriculum no
longer applies).
"""

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp

from hyperleg_rl.assets import HYPERLEG_WO_TOE_CFG

from .hyperleg_env_cfg import HyperLegEnvCfg, HyperLegRewardsCfg
from .mdp import cot_components, heel_grf_magnitude, motor_heat


@configclass
class GotoCommandsCfg:
    """Relative SE(2) pose goal, resampled within the episode."""

    pose_command = nav_mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=nav_mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
    )


@configclass
class GotoObservationsCfg:
    """Same rich state as the walker, with the velocity command replaced by the goal."""

    @configclass
    class PolicyCfg(ObsGroup):
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_effort = ObsTerm(func=mdp.joint_effort)
        last_action = ObsTerm(func=mdp.last_action)
        base_pos_z = ObsTerm(func=mdp.base_pos_z)
        cot_breakdown = ObsTerm(func=cot_components)
        motor_heat_ema = ObsTerm(func=motor_heat)
        heel_grf = ObsTerm(func=heel_grf_magnitude)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        current_time_s = ObsTerm(func=mdp.current_time_s)
        remaining_time_s = ObsTerm(func=mdp.remaining_time_s)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class GotoRewardsCfg(HyperLegRewardsCfg):
    """Reuse termination / power / thermal penalties; swap velocity tracking for position tracking."""

    track_lin_vel_xy_exp: RewTerm | None = None
    track_ang_vel_z_exp: RewTerm | None = None

    position_tracking = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=2.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=2.5,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    heading_tracking = RewTerm(
        func=nav_mdp.heading_command_error_abs,
        weight=-0.5,
        params={"command_name": "pose_command"},
    )


@configclass
class GotoCurriculumCfg:
    """No curriculum (the velocity-command / terrain-level curricula no longer apply)."""

    pass


@configclass
class HyperLegGotoEnvCfg(HyperLegEnvCfg):
    """Toe-equipped end-to-end goal-reaching locomotion env (flat plane)."""

    observations: GotoObservationsCfg = GotoObservationsCfg()
    commands: GotoCommandsCfg = GotoCommandsCfg()
    rewards: GotoRewardsCfg = GotoRewardsCfg()
    curriculum: GotoCurriculumCfg = GotoCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # flat plane (the rough-terrain velocity curriculum is gone)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # space envs out so the ±3 m goals stay inside each env's cell
        self.scene.env_spacing = 8.0


@configclass
class HyperLegGotoEnvCfg_PLAY(HyperLegGotoEnvCfg):
    """Toe-equipped play env (fewer envs, deterministic start, debug markers)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)


@configclass
class HyperLegGotoWoToeEnvCfg(HyperLegGotoEnvCfg):
    """Toe-ablation end-to-end goal-reaching locomotion env."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = HYPERLEG_WO_TOE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class HyperLegGotoWoToeEnvCfg_PLAY(HyperLegGotoEnvCfg_PLAY):
    """Toe-ablation play env."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = HYPERLEG_WO_TOE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
