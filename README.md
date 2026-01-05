## 1-5日 更新 
`问题`：机器人落地垫脚，脚尖先触地，且落地后发生弹跳，使用修正高度后的数据‵kongfan_version2_fix_height3‵
训练命令：python scripts/rsl_rl/train.py \
    --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
    --registry_name weigezhang68-yangzhou-university-org/wandb-registry-motions/kongfan_version2_fix_height \
    --headless \
    --logger wandb \
    --log_project_name BeyondMimic \
    --run_name kongfan_version2_fix_height3 \
    --num_envs 8192 \
    --max_iterations 35000
