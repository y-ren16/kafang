#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1
gpu=(3)
winlens=(1 3 5)
actor_lr=0.0001
seeds=(0)
log_alphas=(0.0 0.005)
gpu_idx=0
mkdir -p ./trainlog_AdjRule
mkdir -p ./trainlog_AdjRule/log_alpha+winlen
mkdir -p ./output_AdjRule
mkdir -p ./output_AdjRule/log_alpha+winlen
for seed in ${seeds[@]}; do
    for log_alpha in ${log_alphas[@]}; do
        for winlen in ${winlens[@]}; do
            CUDA_VISIBLE_DEVICES=${gpu[gpu_idx]} python -u RL_train/SACAdjRule.py --save-dir ./output_AdjRule/log_alpha+winlen/AdjRule_winlen${winlen}_actor-lr${actor_lr}_seed${seed}_log_alpha_${log_alpha} --max-cache-len $winlen --actor-lr $actor_lr --seed $seed --log-alpha $log_alpha > ./trainlog_AdjRule/log_alpha+winlen/log_AdjRule_winlen${winlen}_actor-lr${actor_lr}_seed${seed}_log_alpha_${log_alpha}.txt 2>&1 &
            gpu_idx=$((gpu_idx+1))
            if [ $gpu_idx -ge ${#gpu[@]} ]; then
                gpu_idx=0
            fi
        done
    done
done
wait
