#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1
gpu=(1 2 3 4 5 6)
winlens=(1 3 5)
critic_lr=0.0003
seeds=(0 1)
gpu_idx=0
mkdir -p ./trainlog_3DQN_new
mkdir -p ./trainlog_3DQN_new/seed+winlen
mkdir -p ./output_3DQN_new
mkdir -p ./output_3DQN_new/seed+winlen
for seed in ${seeds[@]}; do
    for winlen in ${winlens[@]}; do
        CUDA_VISIBLE_DEVICES=${gpu[gpu_idx]} python -u RL_train/3DQNTrainerNew.py --save-dir ./output_3DQN_new/seed+winlen/3DQN_new_winlen${winlen}_critic-lr${critic_lr}_seed${seed} --max-cache-len $winlen --critic-lr $critic_lr --seed $seed > ./trainlog_3DQN_new/seed+winlen/log_3dqn_new_winlen${winlen}_critic-lr${critic_lr}_seed${seed}.txt 2>&1 &
        gpu_idx=$((gpu_idx+1))
        if [ $gpu_idx -ge ${#gpu[@]} ]; then
            gpu_idx=0
        fi
    done
done
wait

# CUDA_VISIBLE_DEVICES=7 python RL_train/3DQNTrainerNew.py --save-dir ./output_test/3DQN_new_winlen3_critic-lr0.0003_seed0_20_abs --max-cache-len 3 --critic-lr 0.0003 --seed 0 > ./trainlog_test/log_3dqn_new_winlen3_critic-lr0.0003_seed0_20_abs.txt 2>&1 &