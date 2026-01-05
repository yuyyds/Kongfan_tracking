from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform, sample_log_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# 随机化关节默认位置
def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


# 在给定范围内添加随机值来随机化刚体的质心(CoM)
def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


#########################################
# 随机化刚体质量
def randomize_rigid_body_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the mass of the articulation bodies."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
        
    if asset_cfg.body_ids == slice(None):
        body_ids = slice(None)
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device=asset.device)

    mass = asset.data.default_mass.to(asset.device).clone()
    
    mass = _randomize_prop_by_op(
        mass, mass_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
    )[env_ids]
    
    asset.root_physx_view.set_masses(mass.cpu(), indices=env_ids.cpu())


# 随机化关节的 P (Stiffness) and D (Damping)
# 这比单纯的 Delay 更能模拟真机关节的"软硬"不确定性
def randomize_actuator_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_distribution_params: tuple[float, float],
    damping_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the stiffness and damping of the actuators."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
        
    stiffness = asset.data.default_joint_stiffness.to(asset.device).clone()
    stiffness = _randomize_prop_by_op(
        stiffness, stiffness_distribution_params, env_ids, asset_cfg.joint_ids, operation=operation, distribution=distribution
    )[env_ids]
    
    damping = asset.data.default_joint_damping.to(asset.device).clone()
    damping = _randomize_prop_by_op(
        damping, damping_distribution_params, env_ids, asset_cfg.joint_ids, operation=operation, distribution=distribution
    )[env_ids]

    asset.write_joint_stiffness_to_sim(stiffness, env_ids=env_ids)
    asset.write_joint_damping_to_sim(damping, env_ids=env_ids)


# ----------------------------------------------------------------------
#  针对 Webster Flip 空翻特化的随机化函数
# ----------------------------------------------------------------------
# gpt报错grok改，可以跑
# def randomize_joint_effort_limits(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor | None,
#     asset_cfg: SceneEntityCfg,
#     effort_distribution_params: tuple[float, float],
#     operation: Literal["add", "scale", "abs"] = "scale",
#     distribution: Literal["uniform", "log_uniform"] = "uniform",  # 暂时不支持 gaussian，避免复杂
# ):
#     """
#     随机化关节力矩上限（最稳定版，不依赖有缺陷的 _randomize_prop_by_op）
#     """
#     asset: Articulation = env.scene[asset_cfg.name]
#     device = asset.device

#     if env_ids is None:
#         env_ids = torch.arange(env.num_envs, device=device)

#     # 处理关节选择
#     if asset_cfg.joint_ids == slice(None):
#         joint_ids = slice(None)
#     else:
#         joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.long, device=device)

#     # 读取当前 effort limits（完整 env 部分）
#     current_limits_full = asset.data.joint_effort_limits[env_ids].clone()  # [num_envs, num_joints]

#     # 取出需要随机化的子张量
#     params_to_randomize = current_limits_full if joint_ids == slice(None) else current_limits_full[:, joint_ids]

#     low, high = effort_distribution_params

#     # 手写采样逻辑（使用 Isaac Lab 官方 math 工具，稳定可靠）
#     if distribution == "uniform":
#         noise = sample_uniform(low, high, params_to_randomize.shape, device=device)
#     elif distribution == "log_uniform":
#         noise = sample_log_uniform(low, high, params_to_randomize.shape, device=device)
#     else:
#         raise ValueError(f"Unsupported distribution: {distribution}. Only 'uniform' and 'log_uniform' are supported.")

#     # 应用 operation
#     if operation == "scale":
#         randomized_params = params_to_randomize * noise
#     elif operation == "add":
#         randomized_params = params_to_randomize + noise
#     elif operation == "abs":
#         randomized_params = noise  # abs 操作直接用采样值覆盖
#     else:
#         raise ValueError(f"Unsupported operation: {operation}")

#     # 防止负值或过小
#     randomized_params = torch.clamp(randomized_params, min=1e-3)

#     # 写回完整张量
#     if joint_ids == slice(None):
#         new_limits = randomized_params
#     else:
#         new_limits = current_limits_full.clone()
#         new_limits[:, joint_ids] = randomized_params

#     # 写回缓冲区
#     asset.data.joint_effort_limits[env_ids] = new_limits

#     # 同步到 PhysX
#     asset.write_joint_effort_limit_to_sim(
#         new_limits,
#         joint_ids=joint_ids,
#         env_ids=env_ids,
#     )

# 兼容部分关节随机化
def randomize_joint_effort_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    effort_distribution_params: tuple[float, float],
    operation: Literal["scale", "add"] = "scale",
    distribution: Literal["uniform", "log_uniform"] = "uniform",
):
    """
    随机化关节力矩上限（支持部分关节随机化）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=device)

    # 处理关节选择（关键：使用 SceneEntityCfg 解析后的 joint_ids）
    # 当使用 joint_names 正则时，Isaac Lab 会在内部自动解析成 joint_ids
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.long, device=device)

    # 读取原始值，只取选中关节的部分（形状：[num_envs, num_selected_joints]）
    original_limits = asset.data.joint_effort_limits[env_ids][:, joint_ids].clone()

    low, high = effort_distribution_params

    # 采样噪声（形状自动匹配 original_limits）
    if distribution == "uniform":
        noise = torch.rand_like(original_limits) * (high - low) + low
    elif distribution == "log_uniform":
        log_low = torch.log(torch.tensor(low, device=device))
        log_high = torch.log(torch.tensor(high, device=device))
        noise = torch.exp(torch.rand_like(original_limits) * (log_high - log_low) + log_low)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    # 应用操作
    if operation == "scale":
        new_limits = original_limits * noise
    elif operation == "add":
        new_limits = original_limits + noise
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    new_limits = torch.clamp(new_limits, min=1e-3)

    asset.data.joint_effort_limits[env_ids][:, joint_ids] = new_limits

    asset.write_joint_effort_limit_to_sim(new_limits, joint_ids=joint_ids, env_ids=env_ids)

def push_by_motion_phase(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    command_name: str = "motion",
    push_phases: list[tuple[float, float]] = [(0.0, 0.2), (0.8, 1.0)],   # 默认推起跳前和落地后
):
    """
    根据动作相位施加推力。
    空翻是一个有时序的动作。在空中推它(phase 0.4-0.6)通常没有物理意义(模拟强风除外)。
    我们更希望测试它在"起跳前被推歪"或"落地未稳被推"时的鲁棒性。
    """
    # 获取命令和进度
    command = env.command_manager.get_term(command_name)
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    
    # 计算相位 (0.0 ~ 1.0)
    # command.time_steps: [num_envs]
    # command.motion.time_step_total: scalar
    phases = command.time_steps[env_ids].float() / float(command.motion.time_step_total)
    
    # 筛选需要推的环境 ID，只要相位落在任一区间内，就视为可以推
    push_mask = torch.zeros_like(phases, dtype=torch.bool)
    for (start, end) in push_phases:
        push_mask |= (phases >= start) & (phases <= end)
        
    target_env_ids = env_ids[push_mask]
    
    if len(target_env_ids) == 0:
        return

    # 3. 调用核心推力逻辑 (复用现有的 push_by_setting_velocity 逻辑或重写)
    # 这里我们手动实现一个简单的速度突变
    asset: Articulation = env.scene["robot"]
    
    # 采样随机速度
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    # [len(target_ids), 6]
    vel_noise = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(target_env_ids), 6), device=asset.device)
    
    # 获取当前速度并叠加
    root_vel = asset.data.root_lin_vel_w[target_env_ids]
    root_ang_vel = asset.data.root_ang_vel_w[target_env_ids]
    
    asset.write_root_velocity_to_sim(
        torch.cat([root_vel + vel_noise[:, :3], root_ang_vel + vel_noise[:, 3:]], dim=-1),
        env_ids=target_env_ids
    )
