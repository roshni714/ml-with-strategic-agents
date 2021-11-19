#!/bin/bash
sample_size=(100000 1000000)
for n in "${sample_size[@]}";
do
	name1="derivs_wrt_s"
	name2="derivs_wrt_theta"
	python compute_metrics.py main --n $n --perturbation_s  --save $name1  --partial_deriv_pi_s --partial_deriv_loss_s
	python compute_metrics.py main --n $n --perturbation_theta  --save $name2  --partial_deriv_pi_theta --partial_deriv_loss_theta 
done

