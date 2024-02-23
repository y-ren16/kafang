#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1
gpu=(1 2 3 4 5 6 7)
winlen=5
critic_lr=0.0003
seed=0
gpu_idx=0
betas=(0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001)
mkdir -p ./trainlog_DQN_uncertainty
mkdir -p ./trainlog_DQN_uncertainty/beta
mkdir -p ./output
mkdir -p ./output/dqn_uncertainty_seed
for beta in ${betas[@]}; do
    CUDA_VISIBLE_DEVICES=${gpu[gpu_idx]} python -u RL_train/DQNTrainer_uncertainty.py --save-dir ./output/dqn_uncertainty_seed/DQN_uncertainty_winlen${winlen}_critic-lr${critic_lr}_seed${seed}_beta${beta} --max-cache-len $winlen --critic-lr $critic_lr --seed $seed --beta ${beta} > ./trainlog_DQN_uncertainty/beta/log_dqn_uncertainty_winlen${winlen}_critic-lr${critic_lr}_seed${seed}_beta${beta}.txt 2>&1 &
    gpu_idx=$((gpu_idx+1))
    if [ $gpu_idx -ge ${#gpu[@]} ]; then
        gpu_idx=0
    fi
done
wait

# python RL_train/DQNTrainer_uncertainty.py --save-dir ./output/dqn_uncertainty_seed/DQN_uncertainty_winlen5_critic-lr0.0003_seed0 --max-cache-len 5 --critic-lr 0.0003 --seed 0
# > ./trainlog_DQN_uncertainty/seed/log_dqn_uncertainty_winlen5_critic-lr0.0003_seed0.txt 2>&1 &
