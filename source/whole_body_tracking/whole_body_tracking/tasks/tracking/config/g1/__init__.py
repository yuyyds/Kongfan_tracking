import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-G1-v0",   # 基础的 G1 机器人平面跟踪任务
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-Wo-State-Estimation-v0",   # 不使用状态估计的 G1 平面跟踪任务
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-Low-Freq-v0",  # 低频率控制的 G1 平面跟踪任务
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLowFreqPPORunnerCfg",
    },
)
