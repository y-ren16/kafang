#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1
gpu=(1 2 3 4 5 6)
winlens=(1 3 5)
critic_lr=0.0003
seeds=(0 1)
gpu_idx=0
mkdir -p ./trainlog_DQN
mkdir -p ./trainlog_DQN/seed+winlen
mkdir -p ./output
mkdir -p ./output/dqn_seed+winlen
for seed in ${seeds[@]}; do
    for winlen in ${winlens[@]}; do
        CUDA_VISIBLE_DEVICES=${gpu[gpu_idx]} python -u RL_train/DQNTrainer.py --save-dir ./output/dqn_seed+winlen/DQN_SGD_winlen${winlen}_critic-lr${critic_lr}_seed${seed} --max-cache-len $winlen --critic-lr $critic_lr --seed $seed > ./trainlog_DQN/seed+winlen/log_dqn_SGD_winlen${winlen}_critic-lr${critic_lr}_seed${seed}.txt 2>&1 &
        gpu_idx=$((gpu_idx+1))
        if [ $gpu_idx -ge ${#gpu[@]} ]; then
            gpu_idx=0
        fi
    done
done
wait
