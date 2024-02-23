#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1
gpu=(1 2 3 4 5 6)
winlens=(1 3 5)
actor_lrs=(0.0001)
seeds=(0 1)
gpu_idx=0
mkdir -p ./trainlog_SAC_new
mkdir -p ./trainlog_SAC_new/seed+winlen+fixalpha
mkdir -p ./output_SAC_new
mkdir -p ./output_SAC_new/seed+winlen+fixalpha
for seed in ${seeds[@]}; do
    for actor_lr in ${actor_lrs[@]}; do
        for winlen in ${winlens[@]}; do
            CUDA_VISIBLE_DEVICES=${gpu[gpu_idx]} python -u RL_train/SACtrainer.py --save-dir ./output_SAC_new/seed+winlen+fixalpha/SAC_new_winlen${winlen}_actor-lr${actor_lr}_seed${seed}_alpha-lr0_alpha0.001 --max-cache-len $winlen --actor-lr $actor_lr --seed $seed --alpha-lr 0 --log-alpha -6.907755 > ./trainlog_SAC_new/seed+winlen+fixalpha/log_sac_new_winlen${winlen}_actor-lr${actor_lr}_seed${seed}_alpha-lr0_alpha0.001.txt 2>&1 &
            gpu_idx=$((gpu_idx+1))
            if [ $gpu_idx -ge ${#gpu[@]} ]; then
                gpu_idx=0
            fi
        done
    done
done
wait