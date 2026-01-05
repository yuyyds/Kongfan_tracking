# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")   # 录制视频的长度
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")   # 录制视频的间隔
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wand registry.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)   # 添加 Isaac Sim 模拟器相关的命令行参数

# 两种参数处理机制：
# argparse 处理已知的命令行参数（如 --task, --num_envs 等），已知参数被解析到 args_cli
# Hydra 处理配置相关的参数（通常以 + 或 ~ 开头的参数），未知参数（Hydra 参数）存储在 hydra_args 中
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True  # 模拟器启用相机进行视频录制

# clear out sys.argv for Hydra
# 重新构造 sys.argv，只保留脚本名称和 Hydra 相关的参数，而移除已经被解析的其他参数，防止干扰 Hydra 框架的配置解析
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

# 强化学习环境相关的类和配置
from isaaclab.envs import (
    DirectMARLEnv,   # 多代理强化学习环境
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,    # 多代理环境转单代理环境
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent.(使用 RSL-RL 代理进行训练)"""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.motion_file is None and args_cli.registry_name is None:
        raise ValueError("Either --motion_file or --registry_name must be specified")
    
    if args_cli.motion_file is not None:
        # 从本地加载 motion 文件
        registry_name = args_cli.registry_name = "None"
        env_cfg.commands.motion.motion_file = args_cli.motion_file
        print(f"[INFO] Using local motion file: {args_cli.motion_file}")
    else:
        # load the motion file from the wandb registry
        # 没有指定则默认使用 :latest 版本，若指定则输入 my_model:v1（如model.pt:500?）
        registry_name = args_cli.registry_name
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"
    
        import pathlib
        import wandb
        api = wandb.Api()
        artifact = api.artifact(registry_name)
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
        print(f"[INFO] Using motion file from wandb registry: {registry_name}")
    
    # import pathlib
    # import wandb

    # api = wandb.Api()
    # artifact = api.artifact(registry_name)
    # env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    # specify directory for logging experiments
    # 设置实验日志的根目录，结构为 logs/rsl_rl/{experiment_name}
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # specify directory for logging runs: {time-stamp}_{run_name}
    # 创建运行日志子目录，格式为 {年-月-日_时-分-秒}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    # 使用 gym.make 创建 Isaac Lab 环境，传入任务名称（args_cli.task）和环境配置（env_cfg）。
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    # 检查环境是否为多代理环境,是则使用 multi_agent_to_single_agent 将其转换为单代理环境，以适配 RSL-RL 的要求
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    # 将环境包装为 RslRlVecEnvWrapper，使其兼容 RSL-RL 的向量环境接口，支持并行化训练
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    # write git state to logs
    # 将当前脚本的 Git 仓库状态（版本、提交哈希等）记录到日志中，便于实验可重现性
    runner.add_git_repo_to_log(__file__)
    
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # dump the configuration into log-directory
    # 将环境配置（env_cfg）和代理配置（agent_cfg）保存到日志目录的 params 子目录(保存为 YAML 文件（env.yaml, agent.yaml）和 Pickle 文件（env.pkl, agent.pkl）)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
