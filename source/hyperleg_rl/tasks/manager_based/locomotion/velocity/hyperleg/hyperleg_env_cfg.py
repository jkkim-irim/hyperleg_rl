# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HyperLeg biped locomotion environment configuration.

Single env file: training (`HyperLegEnvCfg`) uses rough terrain, play
(`HyperLegEnvCfg_PLAY`) switches to flat plane. PhysX backend.
"""

import math
from typing import TYPE_CHECKING

import torch

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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor

from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

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
    forces = sensor.data.net_forces_w
    if sensor_cfg.body_ids is not None and sensor_cfg.body_ids != slice(None):
        forces = forces[:, sensor_cfg.body_ids, :]
    return (torch.linalg.norm(forces, dim=-1) > threshold).float()


@configclass
class HyperLegSceneCfg(InteractiveSceneCfg):
    """Training scene for HyperLeg (rough terrain)."""

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
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
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
        resampling_time_range=(8.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 3.0),
            lin_vel_y=(-0.0, 0.0),
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
        scale=1.0,
        alpha=0.4,
    )


@configclass
class HyperLegObservationsCfg:
    """Symmetric observations: actor and critic share the full privileged view.

    Sim2Real is not the target — all privileged signals (base velocities, root
    state, foot contacts, height scan, episode timing) are exposed to the
    policy. No observation noise is injected.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Full-state obs shared by actor and critic (no noise)."""

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_effort = ObsTerm(func=mdp.joint_effort)
        last_action = ObsTerm(func=mdp.last_action)
        base_pos_z = ObsTerm(func=mdp.base_pos_z)
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
class HyperLegRewardsCfg:
    """HyperLeg reward terms."""
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-7)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.015)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["l_tp", "l_hill", "r_tp", "r_hill"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["l_tp", "l_hill", "r_tp", "r_hill"]),
        },
    )
    joint_deviation_hip_yr = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HY", ".*_HR"])},
    )


@configclass
class HyperLegEventsCfg:
    """Domain randomization and resets (PhysX backend)."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 16.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class HyperLegTerminationsCfg:
    """Episode termination terms for HyperLeg."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso"]),
            "threshold": 1.0,
        },
    )


@configclass
class HyperLegCurriculumCfg:
    """Terrain difficulty curriculum."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class HyperLegEnvCfg(ManagerBasedRLEnvCfg):
    """HyperLeg training environment (rough terrain)."""

    scene: HyperLegSceneCfg = HyperLegSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: HyperLegObservationsCfg = HyperLegObservationsCfg()
    actions: HyperLegActionsCfg = HyperLegActionsCfg()
    commands: HyperLegCommandsCfg = HyperLegCommandsCfg()
    rewards: HyperLegRewardsCfg = HyperLegRewardsCfg()
    terminations: HyperLegTerminationsCfg = HyperLegTerminationsCfg()
    events: HyperLegEventsCfg = HyperLegEventsCfg()
    curriculum: HyperLegCurriculumCfg = HyperLegCurriculumCfg()
    sim: SimulationCfg = SimulationCfg(physx=PHYSX_CFG)

    def __post_init__(self):
        self.decimation = 10
        self.episode_length_s = 20.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        elif self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class HyperLegEnvCfg_PLAY(HyperLegEnvCfg):
    """HyperLeg play environment (flat plane)."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # switch to flat plane (overrides training rough terrain)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.commands.base_velocity.ranges.lin_vel_x = (0.7, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
