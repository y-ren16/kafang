#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1
gpu=(0 1 2 3 4 5 6 7)
winlen=1
critic_lrs=(0.00001 0.00005 0.0001 0.0003)
gpu_idx=0
mkdir -p ./trainlog_DQN
mkdir -p ./output
mkdir -p ./output/dqn_lr
for critic_lr in ${critic_lrs[@]}; do
    CUDA_VISIBLE_DEVICES=${gpu[gpu_idx]} python -u RL_train/DQNTrainer.py --save-dir ./output/dqn_lr/DQN_$winlen{winlen}_critic-lr${critic_lr} --max-cache-len $winlen --critic-lr $critic_lr > ./trainlog_DQN/log_dqn_lr_winlen${winlen}_critic-lr${critic_lr}.txt 2>&1 &
    gpu_idx=$((gpu_idx+1))
    if [ $gpu_idx -ge ${#gpu[@]} ]; then
        gpu_idx=0
    fi
done
wait
