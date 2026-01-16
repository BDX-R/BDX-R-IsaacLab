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

def swing_foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    speed_threshold: float,  # We'll use a simple speed check instead of tanh
    std: float,
) -> torch.Tensor:
    """
    Rewards each swinging foot based on how close it is to a target height.
    Feet that are not swinging (i.e., on the ground) receive zero reward.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get the Z-position (height) of the feet
    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    # Get the horizontal (XY) speed of the feet
    foot_velocities_norm = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    # --- NEW, SIMPLER LOGIC ---

    # 1. Identify which feet are swinging
    # A foot is considered "swinging" if its horizontal speed is above a threshold.
    # This results in a tensor of 1s (swinging) and 0s (not swinging).
    is_swinging = (foot_velocities_norm > speed_threshold).float()

    # 2. Calculate the height-based reward for ALL feet
    # This reward is highest (1.0) when the foot is exactly at the target height.
    height_error = torch.square(foot_heights - target_height)
    height_reward = torch.exp(-height_error / std)

    # 3. Apply the reward ONLY to the swinging feet
    # Multiply the height reward by the "is_swinging" mask.
    # If a foot is not swinging, its reward becomes height_reward * 0 = 0.
    reward = height_reward * is_swinging

    # 4. Sum the rewards from all feet for the final result
    return torch.sum(reward, dim=1)

def conditional_joint_deviation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
) -> torch.Tensor:
    """
    Penalizes joint position deviation from the default pose, but ONLY when the
    commanded velocity is below a threshold. This stops fidgeting.
    """
    # 1. Check the command to see if we should be standing still.
    commands = env.command_manager.get_command(command_name)
    command_magnitude = torch.norm(commands, dim=1)
    is_standing_command = (command_magnitude < velocity_threshold)

    # 2. Calculate the joint deviation penalty for ALL environments.
    # This measures how far each joint is from its default "standing" position.
    asset: Articulation = env.scene[asset_cfg.name]
    joint_deviation = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

    # 3. Apply the penalty ONLY to the environments that are commanded to stand still.
    # This is an IF statement: IF is_standing_command is True, THEN penalty = joint_deviation, ELSE penalty = 0.
    penalty = torch.where(is_standing_command, joint_deviation, 0.0)

    return penalty

def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)

def low_speed_sway_penalty(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize linear and angular velocities when command velocity is below threshold."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_speed = torch.norm(command[:, :2], dim=1)
    
    lin_vel_penalty = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    ang_vel_penalty = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    vel_penalty = lin_vel_penalty + ang_vel_penalty
    
    return vel_penalty * (command_speed < command_threshold).float()

# ADD PROVEN PENALTY 2 (The "Limb Guard"):
def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # This uses the helper function we just added
    return joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)

def command_and_contact_gated_foot_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    target_height: float,
    contact_threshold: float,
    std: float,
) -> torch.Tensor:
    """
    Rewards foot clearance ONLY if the robot is commanded to move AND the foot is in the air.
    This is the ultimate version that prevents all known exploits (scraping, wiggling in place).
    """
    # -- GATE 1: Is the robot commanded to move? --
    commands = env.command_manager.get_command(command_name)
    command_magnitude = torch.norm(commands, dim=1)
    # Create a mask: 1.0 if commanded to move, 0.0 if commanded to stand still.
    is_walking_command = (command_magnitude > velocity_threshold).float()

    # If all environments are standing, we can exit early and return zero.
    if torch.all(is_walking_command == 0.0):
        return torch.zeros_like(is_walking_command)

    # -- GATE 2: Is the foot physically in the air? --
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    foot_force_magnitudes = torch.norm(foot_forces, dim=-1)
    is_in_air = (foot_force_magnitudes < contact_threshold).float()

    # -- Calculate the reward --
    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height_error = torch.square(foot_heights - target_height)
    height_reward = torch.exp(-height_error / std)

    # Apply the reward only to feet that are in the air
    clearance_reward_per_foot = height_reward * is_in_air
    # Sum the reward across the feet
    total_clearance_reward = torch.sum(clearance_reward_per_foot, dim=1)

    # -- Final Gating --
    # The final reward is the clearance reward, multiplied by our command gate.
    # If the command was to stand still, this entire reward becomes zero.
    return total_clearance_reward * is_walking_command

def stand_still_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
) -> torch.Tensor:
    """
    Applies a strong penalty for any base motion if the commanded velocity is below a threshold.
    This heavily incentivizes the robot to be perfectly still when commanded to stand.
    """
    # 1. Get the commanded velocity and its magnitude
    commands = env.command_manager.get_command(command_name)
    command_magnitude = torch.norm(commands, dim=1)

    # 2. Identify which environments are commanded to stand still
    is_standing_command = (command_magnitude < velocity_threshold)

    # 3. Get the robot's actual base linear and angular velocity
    asset: RigidObject = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_b
    base_ang_vel = asset.data.root_ang_vel_b

    # 4. Calculate the magnitude of the actual motion
    # We penalize both linear and angular movement.
    motion_magnitude = torch.norm(base_lin_vel, dim=1) + torch.norm(base_ang_vel, dim=1)

    # 5. Apply the penalty ONLY to the environments commanded to stand still
    # The penalty is the magnitude of the motion. If not commanded to stand, penalty is 0.
    penalty = torch.where(is_standing_command, motion_magnitude, 0.0)

    return penalty

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
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)



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
