#!/bin/bash
sample_size=(5000000)
for n in "${sample_size[@]}";
do
	name1="derivs_wrt_s_n_$n"
	name2="derivs_wrt_theta_$n"
	python compute_metrics.py main --n $n --perturbation_s  --save $name1  --partial_deriv_pi_s --partial_deriv_loss_s
	python compute_metrics.py main --n $n --perturbation_theta  --save $name2  --partial_deriv_pi_theta --partial_deriv_loss_theta 
done

