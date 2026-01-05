from isaaclab.utils import configclass  # Isaac Lab 框架提供的装饰器

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"  # 锚点”身体部分为 torso_link（躯干链接）
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",   # 髋关节
            "left_knee_link",   # 膝关节
            "left_ankle_roll_link", # 踝关节
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",   # 躯干
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


# 不包含状态估计(Tracking-Flat-G1-Wo-State-Estimation-v0)
@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


# 低频控制环境配置(Tracking-Flat-G1-Low-Freq-v0)
@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # self.decimation 是环境更新的频率
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)  # 除以低频缩放因子(LOW_FREQ_SCALE)，降低控制频率
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE   # 调整奖励函数中动作变化率的 L2 范数的权重
