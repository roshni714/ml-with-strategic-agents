#!/bin/bash
seeds=(0 1 2 3 4 5 6 7 8 9)
MAX_ITER=100
#for SEED in "${seeds[@]}";
#do
#    python train_beta.py main --n 1000000 --d 2 --perturbation_s 0.2 --perturbation_beta 0.025 --learning_rate 0.25 --max_iter $MAX_ITER --n_types 10 --save dim_2_total --seed $SEED --gradient_type "partial_deriv_loss_beta"
#    python train_beta.py main --n 1000000 --d 2 --perturbation_s 0.2 --perturbation_beta 0.025 --learning_rate 1. --max_iter $MAX_ITER --n_types 10 --save dim_2_total --seed $SEED --gradient_type "total_deriv"

#    python train.py main --n 1000000 --perturbation_s 0.2 --perturbation_theta 0.025 --learning_rate 1. --max_iter $MAX_ITER --n_types 20 --save challenge_n_types_20_n_1000000 --seed $SEED
#    python train.py main --n 1000000 --perturbation_s 0.2 --perturbation_theta 0.025 --learning_rate 1. --max_iter $MAX_ITER --n_types 50 --save challenge_n_types_50_n_1000000 --seed $SEED
#done

for SEED in "${seeds[@]}";
do
    python train_beta.py main --n 1000000 --d 10 --perturbation_s 0.2 --perturbation_beta 0.025 --learning_rate 1. --max_iter $MAX_ITER --n_types 10 --save dim_10 --seed $SEED --gradient_type "total_deriv"
    python train_beta.py main --n 1000000 --d 10 --perturbation_s 0.2 --perturbation_beta 0.025 --learning_rate 1. --max_iter $MAX_ITER --n_types 10 --save dim_10 --seed $SEED --gradient_type "partial_deriv_loss_beta"

#    python train.py main --n 1000000 --perturbation_s 0.2 --perturbation_theta 0.025 --learning_rate 1. --max_iter $MAX_ITER --n_types 20 --save challenge_n_types_20_n_1000000 --seed $SEED
#    python train.py main --n 1000000 --perturbation_s 0.2 --perturbation_theta 0.025 --learning_rate 1. --max_iter $MAX_ITER --n_types 50 --save challenge_n_types_50_n_1000000 --seed $SEED
done


#python train.py main --n 20000 --perturbation_s 0.1 --perturbation_theta 0.1 --learning_rate 0.5 --max_iter 500 --save sim_trial_n_20000
#python train.py main --n 50000 --perturbation_s 0.1 --perturbation_theta 0.1 --learning_rate 0.05 --max_iter 500 --save sim_trial_n_50000
#python train.py main --n 100000 --perturbation_s 0.1 --perturbation_theta 0.1 --learning_rate 0.05 --max_iter 500 --save sim_trial_n_100000
#python train.py main --n 200000 --perturbation_s 0.1 --perturbation_theta 0.1 --learning_rate 0.05 --max_iter 500 --save sim_trial_n_200000



