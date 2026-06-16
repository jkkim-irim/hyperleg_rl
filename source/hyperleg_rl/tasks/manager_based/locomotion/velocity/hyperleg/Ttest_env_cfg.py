# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end point-goal navigation env for HyperLeg ("T test").

A single policy directly controls all 14 leg joints and learns to reach a
commanded position on flat ground using unicycle-style motion: forward speed in
base *x* plus yaw-rate turns. Backward *x* speed above 0.1 m/s or lateral *y*
speed above 1.0 m/s terminates the episode (``termination_penalty`` applies).

When a goal is reached (within ``arrival_std``), a one-shot success bonus is
issued and a new random ``pose_command`` is sampled — the episode continues.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from hyperleg_rl.assets import HYPERLEG_WO_TOE_CFG

from .hyperleg_env_cfg import HyperLegEnvCfg, HyperLegEventsCfg, HyperLegTerminationsCfg
from .mdp import (
    ArrivalResamplePose2dCommandCfg,
    backward_base_lin_vel_x_hinge,
    base_pos_xy,
    cost_of_transport_penalty,
    cot_components,
    goal_arrived_event,
    goal_delta_xy,
    goal_distance_normalized,
    goal_distance_xy,
    goal_pos_xy,
    heel_grf_magnitude,
    inside_arrival_zone,
    lateral_base_lin_vel_y_hinge,
    motor_heat,
    motor_thermal_penalty,
    non_unicycle_base_vel_violation,
    pose_command_progress_reward,
    pose_command_range_levels,
)


@configclass
class TtestActionsCfg:
    """Joint position targets — slightly faster EMA than the velocity pretrain default."""

    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["L_.*", "R_.*"],
        scale=0.5,
        alpha=0.2,
    )


@configclass
class TtestObservationsCfg:
    """Full walker state with the velocity command replaced by the pose goal."""

    @configclass
    class PolicyCfg(ObsGroup):
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        base_pos_xy = ObsTerm(func=base_pos_xy, params={"command_name": "pose_command"})
        goal_pos_xy = ObsTerm(func=goal_pos_xy, params={"command_name": "pose_command"})
        goal_delta_xy = ObsTerm(func=goal_delta_xy, params={"command_name": "pose_command"})
        goal_distance_xy = ObsTerm(func=goal_distance_xy, params={"command_name": "pose_command"})
        goal_distance_normalized = ObsTerm(
            func=goal_distance_normalized, params={"command_name": "pose_command"}
        )
        inside_arrival_zone = ObsTerm(func=inside_arrival_zone, params={"command_name": "pose_command"})
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
        current_time_s = ObsTerm(func=mdp.current_time_s)
        remaining_time_s = ObsTerm(func=mdp.remaining_time_s)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 1.0),
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TtestCommandsCfg:
    """Random position goals; resample on arrival (not on a timer)."""

    pose_command = ArrivalResamplePose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e3, 1.0e3),
        arrival_std=0.1,
        debug_vis=True,
        ranges=ArrivalResamplePose2dCommandCfg.Ranges(
            pos_x=(-1.0, 1.0),
            pos_y=(-1.0, 1.0),
        ),
    )


@configclass
class TtestRewardsCfg:
    """Best distance so far goal progress; unicycle hinge penalties; one-shot goal arrival bonus."""
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-75.0)
    cost_of_transport = RewTerm(
        func=cost_of_transport_penalty,
        weight=-1.0e-4,
        params={"command_name": "pose_command", "velocity_min": 0.1},
    )
    thermal_penalty = RewTerm(
        func=motor_thermal_penalty,
        weight=-5.0,
        params={
            "threshold": {
                "HY": 0.5, "HR": 0.5, "HP": 0.5,
                "KN": 0.5, "AK": 0.5, "FT": 0.5, "TO": 0.5,
            },
        },
    )
    goal_arrived = RewTerm(
        func=goal_arrived_event,
        weight=150.0,
        params={"command_name": "pose_command"},
    )
    position_progress = RewTerm(
        func=pose_command_progress_reward,
        weight=10.0,
        params={"command_name": "pose_command"},
    )

@configclass
class TtestTerminationsCfg(HyperLegTerminationsCfg):
    """Unicycle motion constraint: backward > 0.1 m/s or lateral > 1.0 m/s."""
    bad_orientation = None
    # non_unicycle_motion = DoneTerm(
    #     func=non_unicycle_base_vel_violation,
    #     params={"backward_vel_limit": 0.5, "lateral_vel_limit": 1.0},
    # )


@configclass
class TtestEventsCfg(HyperLegEventsCfg):
    """Flat-plane resets; no mid-episode pushes."""

    push_robot = None

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


@configclass
class TtestCurriculumCfg:
    """Pose-goal range curriculum: ±1 m for 500 iters, then +1 m every 100 iters up to ±10 m."""

    pose_command_range_levels = CurrTerm(
        func=pose_command_range_levels,
        params={
            "command_name": "pose_command",
            "initial_limit": 1.0,
            "final_limit": 10.0,
            "limit_step": 1.0,
            "warmup_iters": 500,
            "iters_per_step": 200,
            "num_steps_per_env": 24,
        },
    )


@configclass
class HyperLegTtestEnvCfg(HyperLegEnvCfg):
    """Toe-equipped end-to-end T-test navigation env (flat plane, from scratch)."""

    observations: TtestObservationsCfg = TtestObservationsCfg()
    actions: TtestActionsCfg = TtestActionsCfg()
    commands: TtestCommandsCfg = TtestCommandsCfg()
    rewards: TtestRewardsCfg = TtestRewardsCfg()
    terminations: TtestTerminationsCfg = TtestTerminationsCfg()
    events: TtestEventsCfg = TtestEventsCfg()
    curriculum: TtestCurriculumCfg = TtestCurriculumCfg()

    def _configure_ttest_common(self, env_spacing: float) -> None:
        """Shared T-test sim / viewer settings (flat and rough variants)."""
        self.scene.terrain.physics_material.static_friction = 2.5
        self.scene.terrain.physics_material.dynamic_friction = 2.0
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.env_spacing = env_spacing
        self.episode_length_s = 30.0
        # Fixed 30 s episodes: timeout is a true terminal (no V bootstrap in RSL-RL).
        self.is_finite_horizon = True
        # Train/video: follow env-0 robot; eye/lookat are offsets from root (world-aligned).
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (5.0, 5.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self._configure_ttest_common(env_spacing=5.0)


@configclass
class HyperLegTtestRoughEnvCfg(HyperLegTtestEnvCfg):
    """Toe-equipped T-test navigation on the same rough terrain as ``HyperLegEnvCfg``."""

    def __post_init__(self):
        # Skip flat-plane override in ``HyperLegTtestEnvCfg``; keep ``HyperLegSceneCfg`` generator terrain.
        super(HyperLegTtestEnvCfg, self).__post_init__()
        self._configure_ttest_common(env_spacing=8.0)
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class TtestPlayCurriculumCfg(TtestCurriculumCfg):
    """Play/eval: fixed full goal range (no curriculum)."""

    pose_command_range_levels = None


@configclass
class HyperLegTtestEnvCfg_PLAY(HyperLegTtestEnvCfg):
    """Toe-equipped play env (fewer envs, deterministic spawn)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.curriculum = TtestPlayCurriculumCfg()
        self.commands.pose_command.ranges.pos_x = (-10.0, 10.0)
        self.commands.pose_command.ranges.pos_y = (-10.0, 10.0)
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)


@configclass
class HyperLegTtestWoToeEnvCfg(HyperLegTtestEnvCfg):
    """Toe-ablation end-to-end T-test navigation env."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = HYPERLEG_WO_TOE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class HyperLegTtestWoToeRoughEnvCfg(HyperLegTtestRoughEnvCfg):
    """Toe-ablation T-test navigation on rough terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = HYPERLEG_WO_TOE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class HyperLegTtestWoToeEnvCfg_PLAY(HyperLegTtestWoToeEnvCfg):
    """Toe-ablation play env."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.curriculum = TtestPlayCurriculumCfg()
        self.commands.pose_command.ranges.pos_x = (-10.0, 10.0)
        self.commands.pose_command.ranges.pos_y = (-10.0, 10.0)
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)