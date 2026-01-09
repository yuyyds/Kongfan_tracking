from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp

##
# Scene definition
##

# VELOCITY_RANGE = {
#     "x": (-0.5, 0.5),
#     "y": (-0.5, 0.5),
#     "z": (-0.2, 0.2),
#     "roll": (-0.52, 0.52),
#     "pitch": (-0.52, 0.52),
#     "yaw": (-0.78, 0.78),
# }

# 推力扰动降低 30%，去除 Z/Roll/Pitch
VELOCITY_RANGE = {
    "x": (-0.35, 0.35),
    "y": (-0.35, 0.35),
    # "z": (-0.14, 0.14),
    # "roll": (-0.364, 0.364),
    # "pitch": (-0.364, 0.364),
    "yaw": (-0.546, 0.546),
}

from isaaclab.terrains.height_field import hf_terrains_cfg
from isaaclab.terrains.sub_terrain_cfg import FlatPatchSamplingCfg

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )

    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


# 强化学习策略(policy)和值函数(critic)的观测空间，分为两个子配置类(PolicyCfg 和 PrivilegedCfg)
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    # 策略网络的观测
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # 获取运动命令
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        # 获取运动锚点,添加均匀噪声 Unoise(-0.25, 0.25) 模拟传感器误差
        ### 若使用 Tracking-Flat-G1-Wo-State-Estimation-v0，则此项为 None
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        # 获取运动锚点的朝向（姿态），添加噪声 Unoise(-0.05, 0.05)
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        ### 若使用 Tracking-Flat-G1-Wo-State-Estimation-v0，则此项为 None
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)  # 上一步的动作（mdp.last_action),无噪声

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # 值函数网络(critic)的观测
    @configclass
    class PrivilegedCfg(ObsGroup):
        # 与 PolicyCfg 类似，但无噪声
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        # body_pos: 获取机器人身体(由 command_name="motion" 指定的身体部分)在基座坐标系中的位置
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        # body_ori: 获取身体部分的朝向
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


# 领域随机化
@configclass
class EventCfg:
    """Configuration for events."""

    # startup(启动时触发)
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # "static_friction_range": (0.3, 1.6),
            # "dynamic_friction_range": (0.3, 1.2),
            "static_friction_range": (0.3, 1.8),
            "dynamic_friction_range": (0.3, 1.5),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # interval(间隔触发)
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )

    
    # [新增] 质量随机化
    mass_randomization = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.85, 1.15), # ±15% 质量误差，这对 Sim2Real 很重要
            "operation": "scale",
        }
    )

    # [新增] 关节刚度/阻尼随机化
    # 模拟电机的响应误差和由于延迟导致的"发软"或"过硬"
    randomize_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2), # P gains 随机化 ±20%
            "damping_distribution_params": (0.8, 1.2),   # D gains 随机化 ±20%
            "operation": "scale",
        },
    )


# 空翻域随机化
@configclass
class EventFlipCfg(EventCfg):
    """
    Configuration for events specifically tuned for acrobatic flips.
    Focus: High impact, actuator saturation, and timing-critical disturbances.
    """
    # -------------------- 物理环境 (Startup) --------------------
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 2.3),
            "dynamic_friction_range": (0.4, 2.0),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 64,
        },
    )

    # -------------------- 机器人动力学 (Startup) --------------------
    # [保留] 质心随机化
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},  # [修改] 质心变化范围
        },
    )
    
    # [增强] 质量随机化
    mass_randomization = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.85, 1.15), # ±15% 质量变化
            "operation": "scale",
        }
    )

    # [增强] 关节刚度/阻尼 (Gains) 随机化
    # 这一点对 Sim2Real 极其重要。空翻时电机处于高频响应区，真实的 PD 响应与仿真差异巨大。
    randomize_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.85, 1.15),  # 从 (0.7, 1.3) 收缩
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    
    # [新增] 关节力矩限制 (Effort Limit) 随机化
    # 模拟电池电压不足或电机性能差异
    # randomize_effort = EventTerm(
    #     func=mdp.randomize_joint_effort_limits,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),   # 显式使用正则匹配所有关节名
    #         "effort_distribution_params": (0.80, 1.0), # 随机只有 80%~100% 的力矩可用
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )
    ##### 分关节版本
    # 下肢（决定起跳）
    randomize_effort_legs = EventTerm(
        func=mdp.randomize_joint_effort_limits,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*hip.*", ".*knee.*", ".*ankle.*"],
            ),
            "effort_distribution_params": (0.85, 1.05),
            "operation": "scale",
        },
    )
    # 上肢（形态为主）
    randomize_effort_arms = EventTerm(
        func=mdp.randomize_joint_effort_limits,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*shoulder.*", ".*elbow.*", ".*wrist.*"],
            ),
            "effort_distribution_params": (0.6, 1.0),
            "operation": "scale",
        },
    )

    # -------------------------------------------------------------------
    # 3. 扰动 (Interval)
    # -------------------------------------------------------------------
    # [替换] 智能相位推力
    # 不再随机推，而是专门挑"起跳前"和"落地后"推
    push_robot = EventTerm(
        func=mdp.push_by_motion_phase,
        mode="interval",
        interval_range_s=(0.5, 1.0), # 检查频率提高，以便捕捉相位
        params={
            "command_name": "motion",
            "push_phases": [(0.0, 0.3), (0.8, 1.0)],
            "velocity_range": {
                "x": (-0.5, 0.5), # 前后推力
                "y": (-0.5, 0.5), # 侧向推力 (对防侧摔很重要)
                "yaw": (-0.5, 0.5), # 旋转扰动
            },
        },
    )


# 奖励函数
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    # motion_body_lin_vel = RewTerm(
    #     func=mdp.motion_global_body_linear_velocity_error_exp,
    #     weight=1.0,
    #     params={"command_name": "motion", "std": 1.0},
    # )
    # motion_body_ang_vel = RewTerm(
    #     func=mdp.motion_global_body_angular_velocity_error_exp,
    #     weight=1.0,
    #     params={"command_name": "motion", "std": 3.14},
    # )
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    # [修改] 降低速度追踪权重 (防止过拟合传感器噪声)
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.5, # 降低权重 (原 1.0)
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.5, # 降低权重 (原 1.0)
        params={"command_name": "motion", "std": 3.14},
    )
    # [修改] 加大动作平滑惩罚 (抑制高频抖动)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.25)  # 加大权重 (原 -0.1)
    # # [新增] 带阈值的脚部接触速度惩罚
    # feet_stumble = RewTerm(
    #     func=mdp.feet_contact_vel_error_with_threshold,
    #     weight=-1.0,  # 强迫机器人"锁死"地面
    #     params={
    #         "command_name": "motion",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
    #         "threshold": 0.1,  # 10cm/s 的死区，允许推力造成的轻微晃动
    #     },
    # )

    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )

# 针对空翻
@configclass
class RewardsFlipCfg(RewardsCfg):
    # ------------------------核心追踪项 (Tracking)------------------------
    # [大幅降低] 全局位置权重（追踪锚点的xyz）
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.2,  # 0.5 -> 0.15
        params={"command_name": "motion", "std": 0.5}, # 增大 std，允许更大的位置误差
    )
    # # 禁用原来的混合追踪,不考虑z轴高度（有问题，会翻过头）
    # motion_global_anchor_pos = None
    # motion_global_anchor_pos_xy = RewTerm(
    #     func=mdp.motion_global_anchor_pos_xy_only_exp,
    #     weight=0.5,  # 高权重希望落点精准
    #     params={"command_name": "motion", "std": 0.5},
    # )
    
    # [大幅增加] 全局姿态权重
    # 空翻的核心是姿态控制，这部分必须严格
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=1.5,   # 0.5 -> 1.5，确保翻转角度精准
        params={"command_name": "motion", "std": 0.2},  # 减小 std，更精确地控制姿态
    )

    # [保留] 身体相对位置 (保持身体形状)
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.3,   # 放松身体姿态追踪,自由去屈膝
        params={"command_name": "motion", "std": 0.5},
    )

    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0, # 恢复权重 (原 1.0)
        params={"command_name": "motion", "std": 1.0},
    )

    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    
    # # [新增] 关键点形状强约束
    # # 强制脚相对于躯干的位置正确，这对起跳和落地姿态至关重要
    # motion_keypoint_shape = RewTerm(
    #     func=mdp.motion_tracking_keypoint_shape_exp,
    #     weight=1.0,  # 约束脚位置稳定
    #     params={
    #         "command_name": "motion", 
    #         "std": 0.5,
    #         # "body_names": ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    #         "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"]
    #     },
    # )

    # # [新增] 起跳脚锁死，防止脚滑
    # takeoff_foot_lock = RewTerm(
    #     func=mdp.takeoff_stance_foot_lock_precise,
    #     weight=-1.5,
    #     params={
    #         "command_name": "motion",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
    #         "threshold": 0.05,
    #         "phase_range": (0.43, 0.54),  # 起跳脚时间窗
    #     },
    # )

    # 奖励机器人在空中达到足够的腾空高度
    base_height = RewTerm(
        func=mdp.base_height_tracking_exp,
        weight=0.5,
        params={
            "command_name": "motion",
            "std": 0.15,
            "min_height": 0.78,
            },
    )

    # # 鼓励双脚在空中滞留，而不是跳得高
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=0.5, # 给予正奖励
    #     params={
    #         "command_name": "motion",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
    #         "threshold": 0.5, # 只要离开地面一段之间
    #     },
    # )

    # ------------------------ 能量与正则化 (Regularization) ------------------------
    # [极度放宽] 动作平滑度
    # 空翻起跳需要极大的瞬时扭矩变化 (Bang-Bang control)。强平滑惩罚会抹平爆发力，导致跳不起来。
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01) # 原 -0.1/-0.25 -> 降至 -0.005

    # [极度降低] 关节位置限制惩罚
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,  # 降低权重，只要不打坏机器人，允许接近极限
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # ------------------------ 落地与接触 (Contact) ------------------------
    # [替换] 使用新的基于参考动作的落地惩罚。
    # 只有当参考动作指示"必须落地"时，才严厉惩罚脚的滑动。在起跳和腾空阶段，允许脚有任何速度
    # feet_stumble = RewTerm(
    #     func=mdp.feet_contact_vel_masked_by_ref,
    #     weight=-0.5,
    #     params={
    #         "command_name": "motion",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
    #         "ref_contact_height_threshold": 0.08,  # 参考脚高小于 8cm 视为接地
    #     },
    # )

    # # 落地瞬间抑制过大的下落速度
    # landing_impact = RewTerm(
    #     func=mdp.landing_impact_penalty,
    #     weight=-1.5,
    #     params={
    #         "command_name": "motion",
    #         "vel_threshold": -1.5,
    #     },
    # )
    # [替换] 落地冲击 -> 改为力峰值惩罚
    landing_shock = RewTerm(
        func=mdp.landing_shock_force_penalty,
        weight=-0.5, # 权重不需要太大，因为它是平方项且数值可能较大
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "threshold_multiplier": 2.0, # 允许瞬间承受 2 倍体重的冲击，超过就开始罚
        },
    )

    # --- [新增] 解决垫脚问题 ---
    # 强力纠正落地脚姿态
    feet_flat_ground = RewTerm(
        func=mdp.feet_flat_ground_orientation_exp,
        weight=1.5, # 给高权重，这直接决定能否站稳
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "std": 0.3,
        }
    )
    # 2. (可选) 如果上面那个不够，再加上这个关节惩罚
    # ankle_reg = RewTerm(
    #    func=mdp.ankle_regularization_landing,
    #    weight=1.0,
    #    params={
    #        "command_name": "motion",
    #        # 注意：这里要填 Joint Name，不是 Body Name。G1 可能是 "left_ankle_pitch"
    #        "joint_names": ["left_ankle_pitch", "right_ankle_pitch"],
    #        "std": 0.5
    #    }
    # )


# 回合终止条件
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # anchor_pos = DoneTerm(
    #     func=mdp.bad_anchor_pos_z_only,
    #     params={"command_name": "motion", "threshold": 0.25},
    # )
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.5},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    # ee_body_pos = DoneTerm(
    #     func=mdp.bad_motion_body_pos_z_only,
    #     params={
    #         "command_name": "motion",
    #         "threshold": 0.25,
    #         "body_names": [
    #             "left_ankle_roll_link",
    #             "right_ankle_roll_link",
    #             "left_wrist_yaw_link",
    #             "right_wrist_yaw_link",
    #         ],
    #     },
    # )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.5,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##
@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)  # 场景配置类
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()   # 观测
    actions: ActionsCfg = ActionsCfg()   # 动作配置类
    commands: CommandsCfg = CommandsCfg()   # 命令配置类
    
    # MDP settings
    # rewards: RewardsCfg = RewardsCfg()  # 奖励（太极）
    rewards: RewardsCfg = RewardsFlipCfg()   # 空翻特化配置

    terminations: TerminationsCfg = TerminationsCfg()  # 终止条件配置类
    
    # events: EventCfg = EventCfg()
    events: EventCfg = EventFlipCfg()   # 空翻特化域随机化

    curriculum: CurriculumCfg = CurriculumCfg()  # 课程学习配置类

    # 后初始化方法，在对象初始化后自动调用
    def __post_init__(self):
        """Post initialization."""
        
        # general settings
        self.decimation = 4   # 环境更新的频率，即每 4 个模拟步执行一次策略（动作更新）
        self.episode_length_s = 10.0
        
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation  # 渲染间隔
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
