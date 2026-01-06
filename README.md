## 1-5日 更新 
`问题`：机器人落地垫脚，脚尖相较于脚跟先触地，且落地后膝盖关节僵直，导致发生弹跳，训练使用修正高度后的数据`kongfan_version2_fix_height3`
## 训练命令：
python scripts/rsl_rl/train.py \
    --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
    --registry_name weigezhang68-yangzhou-university-org/wandb-registry-motions/kongfan_version2_fix_height \
    --headless \
    --logger wandb \
    --log_project_name BeyondMimic \
    --run_name kongfan_version2_fix_height3 \
    --num_envs 8192 \
    --max_iterations 35000

## 1-5日 17.31分更新 
`问题`：我训练了3000step后发现严重问题：我的韦伯斯特空翻是左右脚交替完成的。动作是先右脚撑地，左脚从前往后甩，落地同时替换右脚，然后右脚离空，最后左脚离空完成空翻。我调整了一下takeoff_foot_lock的起跳窗口后让其对应上，训练时在左脚还没落地作为支撑脚时，右脚突然跳了一下，然后右脚在空中时左脚才落地，导致动作不稳。并且修改后的代码训练后机器人空中翻滚落地后左脚没触地，右脚脚尖才够到地，直接向前摔倒，简单来说，翻过头了。
`可能原因`：“翻太高了” $\rightarrow$ “动作做完了人还在天上” $\rightarrow$ “脸着地”，加回z轴约束
## 训练命令：
python scripts/rsl_rl/train.py \
    --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
    --registry_name weigezhang68-yangzhou-university-org/wandb-registry-motions/kongfan_version2_fix_height \
    --headless \
    --logger wandb \
    --log_project_name BeyondMimic \
    --run_name kongfan_version2_fix_height4 \
    --num_envs 8192 \
    --max_iterations 35000

    
## 1-5日 18.40分更新 
`问题`：恢复z轴跟踪，去除`takeoff_foot_lock`与`feet_stumble`奖励，调整权重
## 训练命令：
python scripts/rsl_rl/train.py \
    --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
    --registry_name weigezhang68-yangzhou-university-org/wandb-registry-motions/kongfan_version2_fix_height \
    --headless \
    --logger wandb \
    --log_project_name BeyondMimic \
    --run_name kongfan_version2_fix_height5 \
    --num_envs 8192 \
    --max_iterations 35000
    
