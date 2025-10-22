# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def base_angular_velocity_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, 2]
    ang_vel_error = torch.linalg.norm((target - asset.data.root_ang_vel_b[:, 2]).unsqueeze(1), dim=1)
    return torch.exp(-ang_vel_error / std)


def base_linear_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, ramp_at_vel: float = 1.0, ramp_rate: float = 0.5
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = torch.linalg.norm((target - asset.data.root_lin_vel_b[:, :2]), dim=1)
    # fixed 1.0 multiple for tracking below the ramp_at_vel value, then scale by the rate above
    vel_cmd_magnitude = torch.linalg.norm(target, dim=1)
    velocity_scaling_multiple = torch.clamp(1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0)
    return torch.exp(-lin_vel_error / std) * velocity_scaling_multiple

def bipedal_air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time for a bipedal robot."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if not contact_sensor.cfg.track_air_time:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    # Ensure you are tracking the correct body IDs for the two feet.
    # This will depend on your robot's URDF file.
    # For example, if your foot body IDs are 3 and 6:
    # sensor_cfg.body_ids = [3, 6]

    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)

    # MODIFIED: Expand command to 2 dimensions for the two feet
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 2)

    # MODIFIED: Expand body velocity to 2 dimensions for the two feet
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 2)

    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)

def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    foot_velocities_norm = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    # -- Start: Debugging Print Statement (Re-added) --
    # To avoid flooding the console, we print the heights for only the first environment (env 0).
    # We use .cpu().numpy() to make the tensor printable in a standard format.
    if env.common_step_counter % 10 == 0: # Optional: print only every 10 steps
        print(f"Current foot heights for env 0: {foot_heights[0].cpu().numpy()}")
    # -- End: Debugging Print Statement --

    # --- Original Calculation ---
    foot_z_target_error = torch.square(foot_heights - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * foot_velocities_norm)
    reward_error = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    clearance_reward = torch.exp(-reward_error / std)

    # --- THE FIX: Add a "Walking Gate" ---
    # This gate ensures that the reward is only applied if at least one foot is swinging.
    max_foot_vel = torch.max(foot_velocities_norm, dim=1)[0]
    # The gate is a value from 0 to 1. It's 0 if the robot is still, and 1 if it's walking.
    walking_gate = torch.tanh(5.0 * max_foot_vel)

    # The final reward is the clearance reward scaled by the walking gate.
    return clearance_reward * walking_gate



def bipedal_air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time for a bipedal robot."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if not contact_sensor.cfg.track_air_time:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    # Ensure you are tracking the correct body IDs for the two feet.
    # This will depend on your robot's URDF file.
    # For example, if your foot body IDs are 3 and 6:
    # sensor_cfg.body_ids = [3, 6]

    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)

    # MODIFIED: Expand command to 2 dimensions for the two feet
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 2)

    # MODIFIED: Expand body velocity to 2 dimensions for the two feet
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 2)

    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)

def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_norm = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)

    # Calculate the penalty (deviation from default pose)
    joint_pos_error = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

    # When the command is to stand still, apply a much larger penalty
    # The penalty is negative because we want to discourage this behavior.
    penalty = -joint_pos_error
    scaled_penalty = -stand_still_scale * joint_pos_error

    return torch.where(cmd_norm < velocity_threshold, scaled_penalty, penalty)


# ! look into simplifying the kernel here; it's a little oddly complex
def base_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return 0.8 * torch.square(asset.data.root_lin_vel_b[:, 2]) + 0.2 * torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )


def base_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.projected_gravity_b[:, :2]), dim=1)
