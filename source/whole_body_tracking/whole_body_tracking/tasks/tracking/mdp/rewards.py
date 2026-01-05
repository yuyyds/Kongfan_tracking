from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude
from isaaclab.utils.math import quat_apply

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


# 计算全局锚点位置误差的指数奖励
# 比较命令中的锚点位置与机器人锚点位置,计算平方误差并使用指数函数进行衰减
def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)

# 针对数据悬空改进：忽略 Z 轴的全局位置追踪。解决参考动作悬浮导致机器人“被提着”或者“脚尖够地”的问题
def motion_global_anchor_pos_xy_only_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """
    只追踪 XY 平面位置，完全忽略 Z 轴高度误差。
    允许参考动作飘在空中，而机器人脚踏实地。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # 只取 :2 (x, y)
    target_xy = command.anchor_pos_w[:, :2]
    current_xy = command.robot_anchor_pos_w[:, :2]
    
    error = torch.sum(torch.square(target_xy - current_xy), dim=-1)
    return torch.exp(-error / std**2)


# 计算全局锚点方向误差的指数奖励
# 使用四元数误差度量比较锚点方向,同样使用指数函数进行衰减
def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


# 计算相对身体位置误差的指数奖励
# 只考虑指定的身体部位（通过body_names参数）,计算相对位置误差的均值后再进行指数衰减
def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


# 计算全局身体线性速度误差的指数奖励, 比较身体部位的线性速度误差
def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


# 计算全局身体线性速度误差的指数奖励
def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


# 计算全局身体角速度误差的指数奖励
def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


# 足部接触时间奖励
def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


#########################################################################################
# 带阈值的足部接触速度惩罚
def feet_contact_vel_error_with_threshold(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 0.1
) -> torch.Tensor:
    # 1. 获取机器人对象
    robot: Articulation = env.scene["robot"]
    
    body_ids, _ = robot.find_bodies(sensor_cfg.body_names)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 检查这些特定部位是否接触 (力 > 1.0N)
    is_contact = contact_sensor.data.net_forces_w_history[:, 0, body_ids, 2] > 1.0
    
    # 3. 获取足部在世界坐标系下的线速度 (XY平面)
    foot_vel_xy = robot.data.body_lin_vel_w[:, body_ids, :2]
    
    vel_norm = torch.norm(foot_vel_xy, dim=-1)  # 计算速度模长
    
    vel_error = torch.clamp(vel_norm - threshold, min=0.0)  # 应用死区阈值
    
    reward = is_contact.float() * torch.square(vel_error)   # 计算奖励
    
    return torch.sum(reward, dim=-1)

# 动作加速度惩罚 (平滑二阶导数)
def action_acc_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    计算动作加速度的 L2 范数
    """
    if hasattr(env.action_manager, "prev_action"):
        acc = env.action_manager.action - env.action_manager.prev_action
        return torch.sum(torch.square(acc), dim=1)
    else:
        return torch.zeros(env.num_envs, device=env.device)
    


# ----------------------------------------------------------------------
#  针对 Webster Flip 空翻特化的奖励函数
# ----------------------------------------------------------------------
def feet_contact_vel_masked_by_ref(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg, 
    ref_contact_height_threshold: float = 0.05
) -> torch.Tensor:
    """
    基于参考动作接触状态的落地速度惩罚。
    
    逻辑：
    1. 查参考动作(Command)：如果参考动作的脚高度 < threshold，认为此刻应该落地 (ref_contact = True)。
    2. 如果 ref_contact = True，则计算真机脚部的线速度模长，并给予 L2 惩罚。
    3. 如果 ref_contact = False (在空中)，则奖励为 0 (不惩罚任何速度，允许爆发)。
    
    这对空翻至关重要，因为起跳瞬间脚速极快，不能惩罚。
    """
    # 1. 获取机器人和命令
    robot: Articulation = env.scene["robot"]
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # 获取脚部索引
    body_ids, _ = robot.find_bodies(sensor_cfg.body_names)
    # 获取对应 Command 中的 body index (假设 Command 加载了所有 body，或者需要映射)
    # 注意：这里假设 MotionLoader 加载的 body 顺序包含 sensor_cfg 指定的脚
    cmd_body_indexes = _get_body_indexes(command, sensor_cfg.body_names)

    # 2. 判断参考动作是否接地 (Ref Contact Mask)
    # 获取参考动作中脚的世界坐标 Z 值
    ref_foot_pos_z = command.body_pos_w[:, cmd_body_indexes, 2] 
    # 生成掩码：参考动作脚高低于阈值，视为必须接地
    should_be_contact = ref_foot_pos_z < ref_contact_height_threshold

    # 3. 获取真机脚部 XY 平面速度
    foot_vel_xy = robot.data.body_lin_vel_w[:, body_ids, :2]
    vel_norm_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)

    # 4. 计算惩罚 (只在 should_be_contact 为 True 时惩罚)
    # 使用 float() 将 bool 转换为 0.0/1.0
    reward = should_be_contact.float() * vel_norm_sq
    
    # 对所有脚求和后取负
    return torch.sum(reward, dim=-1)


def motion_tracking_keypoint_shape_exp(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    std: float, 
    body_names: list[str]
) -> torch.Tensor:
    """
    关键点形状追踪奖励。
    不同于全局位置追踪，这个奖励关注"脚相对于基座(Base)"的位置。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene["robot"]
    
    # 获取索引
    cmd_body_indexes = _get_body_indexes(command, body_names)
    robot_body_indexes, _ = robot.find_bodies(body_names)
    
    # --- 计算参考动作的相对位置 (Ref Body - Ref Base) ---
    # Command 中已有 body_pos_relative_w (这是相对于 Anchor 的)，如果 Anchor 是 Base/Torso 则直接用
    # 假设 anchor_body_name 是 torso/base
    ref_rel_pos = command.body_pos_relative_w[:, cmd_body_indexes]

    # --- 计算真机的相对位置 (Real Body - Real Base) ---
    # 获取机器人基座位置 (假设 Base 是索引 0，或者通过 robot.data.root_pos_w)
    # 更严谨的做法是获取 Anchor 的索引
    anchor_idx = robot.body_names.index(command.cfg.anchor_body_name)
    
    robot_body_pos = robot.data.body_pos_w[:, robot_body_indexes]
    robot_anchor_pos = robot.data.body_pos_w[:, anchor_idx].unsqueeze(1) # [Env, 1, 3]
    
    # 简单的向量差 (在世界坐标系下的相对向量)
    # 注意：为了更严格，应该转换到 Base 的局部坐标系，但在追踪任务中，
    # 只要 Base Orientation 追踪得好，世界系下的相对向量差也是有效的
    robot_rel_pos = robot_body_pos - robot_anchor_pos
    
    # 计算误差
    error = torch.sum(torch.square(ref_rel_pos - robot_rel_pos), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def takeoff_stance_foot_lock_precise(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.05,
    phase_range: tuple[float, float] = (0.0, 0.3),  # 脚的时间窗
) -> torch.Tensor:
    """
    基于相位和接触力的稳健起跳脚防滑奖励
    """
    robot: Articulation = env.scene["robot"]
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    phase = command.time_steps.float() / float(command.motion.time_step_total)
    
    # 使用制动起跳精确范围
    takeoff_phase_mask = (phase >= phase_range[0]) & (phase <= phase_range[1])

    # 接触力判断 (> 30N 视为支撑)
    body_ids, _ = robot.find_bodies(sensor_cfg.body_names)
    foot_force_z = contact_sensor.data.net_forces_w_history[:, 0, body_ids, 2]
    is_bearing_load = foot_force_z > 30.0

    # 计算惩罚
    foot_vel_xy = robot.data.body_lin_vel_w[:, body_ids, :2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)
    
    # 使用 ReLU (clamp) 计算超过阈值的部分
    vel_error = torch.clamp(vel_norm - threshold, min=0.0)

    # 组合 Mask
    active_mask = takeoff_phase_mask.unsqueeze(1) * is_bearing_load.float()
    
    reward = torch.sum(active_mask * torch.square(vel_error), dim=-1)   # 惩罚
    
    return reward

def base_height_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    min_height: float = 0.05,
):
    """
    奖励机器人在空中达到足够的腾空高度(只在脚离地后生效,防止策略学成“贴地翻”)
    """
    robot: Articulation = env.scene["robot"]
    command: MotionCommand = env.command_manager.get_term(command_name)

    # base / anchor 高度
    base_z = robot.data.root_pos_w[:, 2]
    ref_base_z = command.anchor_pos_w[:, 2]

    # 是否已经离地（参考动作）
    in_air = ref_base_z > min_height

    error = torch.square(base_z - ref_base_z)
    reward = torch.exp(-error / std**2)

    return reward * in_air.float()

def landing_impact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    vel_threshold: float = -2.0,  # 垂直速度的下限阈值(2 m/s)
):
    """
    落地瞬间抑制过大的下落速度
    """
    robot: Articulation = env.scene["robot"]
    command: MotionCommand = env.command_manager.get_term(command_name)

    phase = command.time_steps.float() / float(command.motion.time_step_total)
    landing_mask = phase > 0.60

    # 机器人根部的世界坐标系垂直速度（Z 轴）
    vz = robot.data.root_lin_vel_w[:, 2]  # 向下为负
    penalty = torch.clamp(-(vz - vel_threshold), min=0.0)

    return penalty * landing_mask.float()

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
    奖励双脚在空中的时间。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]   # 获取传感器
    
    robot: Articulation = env.scene["robot"]   # 获取机器人句柄（robot）
    
    body_ids, _ = robot.find_bodies(sensor_cfg.body_names)  # 查找脚的索引
    
    # 获取滞空时间
    # contact_sensor.data.current_air_time 记录了每个刚体连续未接触地面的时间(秒)
    air_time = contact_sensor.data.current_air_time[:, body_ids]
    
    # 只要在空中就给分，但设置一个上限 threshold (比如 0.5秒)，防止为了刷分一直不落地
    reward = torch.clamp(air_time, max=threshold)   # 计算奖励
    
    return torch.sum(reward, dim=1)  # 奖励双脚同时离地

# 落地强制脚掌放平 (Foot Flat Orientation)
def feet_flat_ground_orientation_exp(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg, 
    std: float = 0.5
) -> torch.Tensor:
    """
    在落地阶段，奖励脚掌与地面平行（防止垫脚或脚跟单独着地）
    """
    robot: Articulation = env.scene["robot"]
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # 计算相位，只在动作后段生效
    phase = command.time_steps.float() / float(command.motion.time_step_total)
    landing_mask = phase > 0.62

    # 获取脚部刚体的四元数
    body_ids, _ = robot.find_bodies(sensor_cfg.body_names)
    foot_quat_w = robot.data.body_quat_w[:, body_ids]  # [Env, 2, 4] (假设双脚)
    
    # 定义局部 Z 轴向量
    local_z = torch.zeros((env.num_envs, len(body_ids), 3), device=env.device)
    local_z[..., 2] = 1.0 
    
    # 旋转到世界系
    world_z_from_foot = quat_apply(foot_quat_w, local_z) # [Env, 2, 3]
    
    # 4. 计算误差：我们希望 world_z_from_foot 的 Z 分量接近 1.0 (完全垂直向上)
    # 误差 = 1.0 - z_component
    alignment_error = 1.0 - world_z_from_foot[..., 2] # 越接近 0 越好
    
    rew = torch.exp(-alignment_error / std**2)  # 计算奖励
    
    # 对双脚取平均，并乘以 mask
    return torch.sum(rew, dim=-1) * landing_mask.float()

# 踝关节角度正则化
def ankle_regularization_landing(
    env: ManagerBasedRLEnv,
    command_name: str,
    joint_names: list[str], # 传入 ankle_pitch 相关的关节名
    std: float = 0.5
) -> torch.Tensor:
    """
    落地阶段，惩罚踝关节角度过大（防止垫脚）。
    """
    robot: Articulation = env.scene["robot"]
    command: MotionCommand = env.command_manager.get_term(command_name)

    phase = command.time_steps.float() / float(command.motion.time_step_total)
    landing_mask = phase > 0.62
    
    # 获取关节位置
    joint_ids, _ = robot.find_joints(joint_names)
    # 获取关节角度 (相对于默认位置的差值，或者绝对值，视你的观测而定，这里取绝对位置)
    joint_pos = robot.data.joint_pos[:, joint_ids]
    
    # 计算误差
    error = torch.sum(torch.square(joint_pos), dim=-1)
    
    return torch.exp(-error / std**2) * landing_mask.float()