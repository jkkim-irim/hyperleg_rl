# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HyperLeg biped locomotion environment configuration (rough terrain).

Uses the PhysX physics backend.
"""

import math
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg

if TYPE_CHECKING:
    from isaaclab_physx.sensors import ContactSensor

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from hyperleg_rl.assets import HYPERLEG_CFG

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


PHYSX_CFG = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)


def feet_contact_bool(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Per-body contact state as 0.0/1.0 floats.

    Args:
        env: env instance.
        sensor_cfg: contact sensor entity with ``body_names`` selecting the bodies to query.
        threshold: contact-force magnitude threshold [N]. Defaults to 1.0.

    Returns:
        Tensor of shape ``[num_envs, num_bodies]`` with 1.0 if ``||net_force|| > threshold``
        else 0.0.
    """
    sensor: "ContactSensor" = env.scene.sensors[sensor_cfg.name]
    forces = wp.to_torch(sensor.data.net_forces_w)
    if sensor_cfg.body_ids is not None and sensor_cfg.body_ids != slice(None):
        forces = forces[:, sensor_cfg.body_ids, :]
    return (torch.linalg.norm(forces, dim=-1) > threshold).float()


@configclass
class HyperLegSceneCfg(InteractiveSceneCfg):
    """Rough-terrain scene for HyperLeg."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot = HYPERLEG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class HyperLegCommandsCfg:
    """Velocity commands for HyperLeg."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class HyperLegActionsCfg:
    """Joint position targets for all 14 leg DoFs."""

    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["L_.*", "R_.*"],
        scale=0.5,
        alpha=1.0,
    )


@configclass
class HyperLegObservationsCfg:
    """Asymmetric actor-critic observations.

    Actor (``policy``) sees only proprioceptive signals that map to real-robot
    sensors (IMU + joint encoder + prior action + velocity command). Critic
    (``privileged``) sees the full simulator state plus height scan and
    per-foot contact bools.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor obs: deployable on real robot (IMU + encoder + command + last action)."""

        last_action = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Critic-only obs: full sim state (no noise) + height scan + foot contact bools."""

        # Mirror of actor view (clean, no noise)
        last_action = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_effort = ObsTerm(func=mdp.joint_effort)
        # Root state (state estimation required on real robot → privileged)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_pos_z = ObsTerm(func=mdp.base_pos_z)
        root_pos_w = ObsTerm(func=mdp.root_pos_w)
        root_quat_w = ObsTerm(func=mdp.root_quat_w)
        root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w)
        root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w)
        # Per-foot contact bools, ordered [lft, lto, rft, rto]
        feet_contact = ObsTerm(
            func=feet_contact_bool,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[".*lft$", ".*lto$", ".*rft$", ".*rto$"]
                ),
                "threshold": 1.0,
            },
        )
        # Episode timing
        current_time_s = ObsTerm(func=mdp.current_time_s)
        remaining_time_s = ObsTerm(func=mdp.remaining_time_s)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class HyperLegRewardsCfg:
    """HyperLeg rough-terrain reward terms."""
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-6)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.2)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.2)},
    )
    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-7)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.015)


@configclass
class HyperLegEventsCfg:
    """Domain randomization and resets (PhysX backend)."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*pelvis$"]),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*pelvis$"]),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}},
    )


@configclass
class HyperLegTerminationsCfg:
    """Episode termination terms for HyperLeg."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    pelvis_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*pelvis$"]),
            "threshold": 1.0,
        },
    )


@configclass
class HyperLegCurriculumCfg:
    """Terrain difficulty curriculum."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class HyperLegRoughEnvCfg(ManagerBasedRLEnvCfg):
    """HyperLeg rough environment configuration."""

    scene: HyperLegSceneCfg = HyperLegSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: HyperLegObservationsCfg = HyperLegObservationsCfg()
    actions: HyperLegActionsCfg = HyperLegActionsCfg()
    commands: HyperLegCommandsCfg = HyperLegCommandsCfg()
    rewards: HyperLegRewardsCfg = HyperLegRewardsCfg()
    terminations: HyperLegTerminationsCfg = HyperLegTerminationsCfg()
    events: HyperLegEventsCfg = HyperLegEventsCfg()
    curriculum: HyperLegCurriculumCfg = HyperLegCurriculumCfg()
    sim: SimulationCfg = SimulationCfg(physics=PHYSX_CFG)

    def __post_init__(self):
        self.decimation = 20
        self.episode_length_s = 20.0
        self.sim.dt = 0.001
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        elif self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class HyperLegRoughEnvCfg_PLAY(HyperLegRoughEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.7, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
