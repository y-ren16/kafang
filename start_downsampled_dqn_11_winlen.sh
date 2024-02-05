#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1
gpu=(7 6 5 4 3 2 1 0)
winlens=(1 2 3 4 5)
critic_lr=0.0003
gpu_idx=0
mkdir -p ./trainlog_DQN
mkdir -p ./output
mkdir -p ./output/dqn_lr
for winlen in ${winlens[@]}; do
    CUDA_VISIBLE_DEVICES=${gpu[gpu_idx]} python -u RL_train/DQNTrainer.py --save-dir ./output/dqn_win/DQN_winlen${winlen}_critic-lr${critic_lr} --max-cache-len $winlen --critic-lr $critic_lr > ./trainlog_DQN/log_dqn_win_winlen${winlen}_critic-lr${critic_lr}.txt 2>&1 &
    gpu_idx=$((gpu_idx+1))
    if [ $gpu_idx -ge ${#gpu[@]} ]; then
        gpu_idx=0
    fi
done
wait
