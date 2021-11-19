#!/bin/bash
python train.py main --n 100 --perturbation_s 0.1 --perturbation_theta 0.1 --learning_rate 0.05 --max_iter 5 --save test_run
python train.py main --n 20000 --perturbation_s 0.1 --perturbation_theta 0.1 --learning_rate 0.05 --max_iter 500 --save sim_trial_n_20000
python train.py main --n 50000 --perturbation_s 0.1 --perturbation_theta 0.1 --learning_rate 0.05 --max_iter 500 --save sim_trial_n_50000
python train.py main --n 100000 --perturbation_s 0.1 --perturbation_theta 0.1 --learning_rate 0.05 --max_iter 500 --save sim_trial_n_100000
python train.py main --n 200000 --perturbation_s 0.1 --perturbation_theta 0.1 --learning_rate 0.05 --max_iter 500 --save sim_trial_n_200000



