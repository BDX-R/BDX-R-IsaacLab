# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ImuCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
#from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv  # <<<<<<<<<< MOVE THE IMPORT HERE
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    # It's fine to keep this import here for type checkers if needed, but the one above is essential for runtime.
    from isaaclab.managers import RewardTermCfg

# TODO: Clean up all non used imports

##
# Pre-defined configs
##

from BDXR.robots.bdxr import BDX_CFG  # isort:skip

import BDXR.tasks.bdxr_locomotion.mdp as mdp  # isort:skip

# Old Version
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),
    },
)

#COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
#    size=(8.0, 8.0),
#    border_width=20.0,
#    num_rows=9,
#    num_cols=21,
#    horizontal_scale=0.1,
#    vertical_scale=0.002,  # This is already quite low, which is good for an easier terrain
#    slope_threshold=0.75,
#    difficulty_range=(0.0, 1.0),
#    use_cache=False,
#    sub_terrains={
#        # Increase the proportion of flat ground
#        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.8),
#        # Decrease the proportion of rough ground
#        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
#            proportion=0.2,
#            noise_range=(0.02, 0.05),
#            noise_step=0.02,
#            border_width=0.25
#        ),
#    },
#)

# TODO: Are all critics necessary (i don't think so) 
# TODO: In critic we should remove all noise
# TODO: Add some scales to avoid sim instabilities
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class CriticCfg(ObsGroup):
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        actions = ObsTerm(func=mdp.last_action)
        imu_projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            scale =0.2,
        )
        imu_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )

        # Privileged Critic Observations
        # Joint dynamics information (privileged)
        joint_torques = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )

        # Contact forces on feet (privileged foot contact information)
        feet_contact_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=[".*_Foot"]
                )
            },
        )

        # Body poses for important body parts (privileged state info)
        body_poses = ObsTerm(
            func=mdp.body_pose_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[".*_Foot"],
                )
            },
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )

        # Joint positions and velocities with less noise (privileged accurate state)
        joint_pos_accurate = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )
        joint_vel_accurate = ObsTerm(
            func=mdp.joint_vel_rel,
            scale =0.05,
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )

        # Base position (full pose information - privileged)
        base_pos = ObsTerm(
            func=mdp.base_pos_z, noise=Unoise(n_min=-0.0001, n_max=0.0001)
        )

        # Root state information (privileged)
        root_lin_vel_w = ObsTerm(
            func=mdp.root_lin_vel_w, noise=Unoise(n_min=-0.0001, n_max=0.0001)
        )
        root_ang_vel_w = ObsTerm(
            func=mdp.root_ang_vel_w, noise=Unoise(n_min=-0.0001, n_max=0.0001)
        )

        # No noise for the critic
        def __post_init__(self):
            self.enable_corruption = False

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel, scale=0.2, noise=Unoise(n_min=-0.35, n_max=0.35))

        # TODO: Adds IMu Projected Gravity
        imu_projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Observation groups:
    critic: CriticCfg = CriticCfg()
    policy: PolicyCfg = PolicyCfg()

# TODO: if we change almost anything, we should redo it from scratch and not base it on rewardcfg
@configclass
class BDXRRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1, params={"command_name": "base_velocity", "std": 0.5}
    )
    #base_linear_velocity = RewTerm(
    #    func=mdp.base_linear_velocity_reward,
    #    weight=5.0,
    #    params={"std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    #)
    #base_angular_velocity = RewTerm(
    #    func=mdp.base_angular_velocity_reward,
    #    weight=2.5,
    #    params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    #)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot"),
            "threshold": 0.2,
        },
    )
    air_time = RewTerm(
        func=mdp.bipedal_air_time_reward,
        weight=1.0,
        params={
            "mode_time": 0.5,
            "velocity_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot"),
        },
    )
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-1.0, # Or -5.0 if you use the original function
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,  # <-- Increase this value substantially
            "velocity_threshold": 0.3,
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Yaw", ".*_Hip_Roll"])},
    )
    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Neck_Pitch", "Head_Pitch", "Head_Yaw", "Head_Roll"])},
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle"])},
    )
    sway_penalty = RewTerm(
        func=mdp.low_speed_sway_penalty,
        weight=-5.0, # A huge penalty for any torso movement when commanded to be still.
        params={ "command_name": "base_velocity", "command_threshold": 0.1 }
    )
    fidget_penalty = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-5.0,  # A strong penalty for any joint fidgeting when commanded to be still.
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        }
    )
    foot_clearance = RewTerm(
        func=mdp.command_and_contact_gated_foot_clearance,
        weight=5.0,
        params={
            # NO resolve_ids=True here. The framework does it automatically.
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot"),

            "command_name": "base_velocity",
            "velocity_threshold": 0.05,
            "target_height": 0.15,
            "contact_threshold": 1.0,
            "std": 0.05,
        },
    )
    base_motion = RewTerm(
        func=mdp.base_motion_penalty, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_orientation = RewTerm(
        func=mdp.base_orientation_penalty, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    rhythm_penalty = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,  # Start with a moderate penalty
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot"),
        }
    )
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-1, # Start with -0.5. Since it uses squared values, this is quite strong.
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
# TODO: With the way it's implemented, it is not used, should we remove it or use itS
@configclass
class EventCfg:
    """Configuration for events."""
    # startup
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
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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
#
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)}, # 0.02
        },
    )

    

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link","LEFT_UPPER_LEG"]), "threshold": 1.0},
    )
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.25})
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, 
        params={
            "limit_angle": math.radians(60.0),
        },
    )
# TODO: clean this up
@configclass
class BdxrEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: BDXRRewards = BDXRRewards()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = BDX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        
        # CORRECT IMU CONFIGURATION
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",  # Attach to the actual rigid body
            debug_vis=False,
            offset=ImuCfg.OffsetCfg(  # Use the nested OffsetCfg class
                pos=(-0.04718, 0.0663, 0.11094),    # From your screenshot
                rot=(1.0, 0.0, 0.0, 0.0),           # (w, x, y, z) for no rotation
            )
        )
        # actions
        self.actions.joint_pos.scale = 0.25

        # events
        self.events.push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
        #self.events.push_robot = None
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 0.5)
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.6)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.2)
        self.events.physics_material.params["asset_cfg"].body_names = ".*_Foot"
        # terrain
        # TODO: This does not need to be fully placed, we can do the same in 3 line
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        # Randomization
        #self.events.push_robot = None
        #self.events.add_base_mass = None
        #self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # Rewards
        # TODO: remove all this to place it in the reward manager directly
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -5
        self.rewards.action_rate_l2.weight = -0.5
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_Hip_.*", ".*_Knee"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_Hip_.*", ".*_Knee", ".*_Ankle.*"]
        )
        #self.rewards.track_lin_vel_xy_exp = None
        #self.rewards.track_ang_vel_z_exp = None

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.1

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link", "Left_Lower_Leg", "Right_Lower_Leg", "Right_Hip_Pitch_Motor", "Left_Hip_Pitch_Motor"]


@configclass
class BdxrEnvCfg_PLAY(BdxrEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # remove random pushing event