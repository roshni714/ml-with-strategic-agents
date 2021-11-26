#!/bin/bash
perturbation_s=(0.2)
perturbation_theta=(0.025)
sample_size=(1000000)
n_types_num=(10 20 50)
for PERTURB_S in "${perturbation_s[@]}";
do
	for PERTURB_THETA in "${perturbation_theta[@]}";
	do
		for N in "${sample_size[@]}";
		do
			for N_TYPES in "${n_types_num[@]}";
			do
 			    str_perturb_s="$PERTURB_S"
			    str_perturb_theta="$PERTURB_THETA"
			    replace_dot_theta=${str_perturb_theta//./_}
			    replace_dot=${str_perturb_s//./_}
			    perturb_s="perturb_s_${replace_dot}_perturb_theta_${replace_dot_theta}" 
			    sample="_n_$N"
			    n_types_val="_ntypes_$N_TYPES"
			    name="${perturb_s}${sample}${n_types_val}"
			    echo $name
			    python plot_derivative.py main --n $N --n_types $N_TYPES --perturbation_s $PERTURB_S --perturbation_theta $PERTURB_THETA --save $name  --total_deriv --partial_deriv_loss_theta --partial_deriv_pi_theta --partial_deriv_pi_s --partial_deriv_loss_s --partial_deriv_s_theta --density
		    	done
		done
	done
done

#	python plot_derivative.py main --n 20000 --perturbation_s 0.05 --perturbation_theta 0.01 --save sim_trial_perturbation_s_0_05 --total_deriv --partial_deriv_loss_theta --partial_deriv_pi_theta --partial_deriv_pi_s --partial_deriv_loss_s --partial_deriv_s_theta --density
#	python plot_derivative.py main --n 50000 --perturbation_s 0.05 --perturbation_theta 0.01 --save sim_trial_perturbation_s_0_05 --total_deriv --partial_deriv_loss_theta --partial_deriv_pi_theta --partial_deriv_pi_s --partial_deriv_loss_s --partial_deriv_s_theta --density
#	python plot_derivative.py main --n 100000 --perturbation_s 0.05 --perturbation_theta 0.01 --save sim_trial_perturbation_s_0_05 --total_deriv --partial_deriv_loss_theta --partial_deriv_pi_theta --partial_deriv_pi_s --partial_deriv_loss_s --partial_deriv_s_theta --density
#	python plot_derivative.py main --n 200000 --perturbation_s 0.05 --perturbation_theta 0.01 --save sim_trial_perturbation_s_0_05 --total_deriv --partial_deriv_loss_theta --partial_deriv_pi_theta --partial_deriv_pi_s --partial_deriv_loss_s --partial_deriv_s_theta --density

