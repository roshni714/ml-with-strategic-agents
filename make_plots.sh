#!/bin/bash
python plot_total_derivative.py main --n 100 --perturbation_s 0.1 --perturbation_theta 0.1 --save test_run_perturbation_0_1
python plot_total_derivative.py main --n 20000 --perturbation_s 0.1 --perturbation_theta 0.1 --save sim_trial_n_20000_perturbation_0_1
python plot_total_derivative.py main --n 50000 --perturbation_s 0.1 --perturbation_theta 0.1 --save sim_trial_n_50000_perturbation_0_1
python plot_total_derivative.py main --n 100000 --perturbation_s 0.1 --perturbation_theta 0.1 --save sim_trial_n_100000_perturbation_0_1
python plot_total_derivative.py main --n 200000 --perturbation_s 0.1 --perturbation_theta 0.1 --save sim_trial_n_200000_perturbation_0_1



