import numpy as np
import argh
import matplotlib.pyplot as plt

from agent_distribution import AgentDistribution
from gradient_estimation import GradientEstimator
from utils import compute_continuity_noise, fixed_point_interpolation_true_distribution
from reparametrized_gradient import expected_total_derivative
from expected_gradient import ExpectedGradient


def plot(
    agent_dist,
    sigma,
    q,
    f,
    derivatives_to_plot,
    perturbation_s_size=0.1,
    perturbation_theta_size=0.1,
    true_beta=None,
):
    derivs_exp_all = []
    derivs_emp_all = []
    thetas = np.linspace(-np.pi, np.pi, 50)
    for theta in thetas:
        s = f(theta)
        grad_exp = ExpectedGradient(agent_dist, theta, s, sigma, true_beta)
        val = grad_exp.compute_total_derivative()
        grad_est = GradientEstimator(
            agent_dist,
            theta,
            s,
            sigma,
            q,
            true_beta,
            perturbation_s_size=perturbation_s_size,
            perturbation_theta_size=perturbation_theta_size,
        )
        hat_val = grad_est.compute_total_derivative()

        derivs_exp_all.append(val)
        derivs_emp_all.append(hat_val)

    fig, ax = plt.subplots(
        1, len(derivatives_to_plot), figsize=(5, (12 / 5) * len(derivatives_to_plot))
    )
    
    if len(derivatives_to_plot) > 1:
        for i in range(len(derivatives_to_plot)):
            deriv_emp = [dic[derivatives_to_plot[i]] for dic in derivs_emp_all]
            deriv_exp = [dic[derivatives_to_plot[i]] for dic in derivs_exp_all]

            ax[i].plot(thetas, deriv_emp, label="empirical")
            ax[i].plot(thetas, deriv_exp, label="expectation")
            ax[i].set_xlabel("theta")
            ax[i].set_ylabel(derivatives_to_plot[i])
            ax[i].set_title("theta vs. {}".format(derivatives_to_plot[i]))
    else:
        deriv_emp = [dic[derivatives_to_plot[i]] for dic in derivs_emp_all]
        deriv_exp = [dic[derivatives_to_plot[i]] for dic in derivs_exp_all]

        plt.plot(thetas, deriv_emp, label="empirical")
        plt.plot(thetas, deriv_exp, label="expectation")
        plt.xlabel("theta")
        plt.ylabel(derivatives_to_plot[i])
    if savefig is not None:
        title = savefig.split("/")[-1]
        plt.suptitle(title)
        plt.savefig(savefig)
    plt.show()
    plt.close()

@argh.arg("--n", default=100000)
@argh.arg("--perturbation_s", default=0.1)
@argh.arg("--perturbation_theta", default=0.1)
@argh.arg("--total_deriv", default=False)
@argh.arg("--partial_deriv_loss_theta", default=False)
@argh.arg("--partial_deriv_pi_theta", default=False)
@argh.arg("--partial_deriv_pi_s", default=False)
@argh.arg("--partial_deriv_loss_s", default=False)
@argh.arg("--partial_deriv_s_theta", default=False)
def main(
    n=100000,
    perturbation_s=0.1,
    perturbation_theta=0.1,
    save="results",
    total_deriv=True,
    partial_deriv_loss_theta=True,
    partial_deriv_pi_theta=True,
    partial_deriv_pi_s=True,
    partial_deriv_loss_s=True,
    partial_deriv_s_theta=True,
):
    np.random.seed(0)

    n_types = 1
    d = 2
    etas = np.random.uniform(0.3, 0.8, n_types * d).reshape(n_types, d, 1)
    gammas = np.random.uniform(5.0, 8.0, n_types * d).reshape(n_types, d, 1)
    dic = {"etas": etas, "gammas": gammas}
    agent_dist = AgentDistribution(n=n, d=d, n_types=n_types, types=dic, prop=None)
    #    sigma = compute_continuity_noise(agent_dist)
    true_beta = np.array([1., 0.]).reshape(2, 1)

    derivatives_to_plot = []
    if total_deriv:
        derivatives_to_plot.append("total_deriv")
    if partial_deriv_loss_theta:
        derivatives_to_plot.append("partial_deriv_loss_theta")
    if partial_deriv_pi_theta:
        derivatives_to_plot.append("partial_deriv_pi_theta")
    if partial_deriv_pi_s:
        derivatives_to_plot.append("partial_deriv_loss_s")
    if partial_deriv_s_theta:
        derivatives_to_plot.append("partial_deriv_s_theta")

    sigma = 0.35
    q = 0.7
    f = fixed_point_interpolation_true_distribution(
        agent_dist, sigma, q, plot=False, savefig=None
    )
    plot(
        agent_dist,
        sigma,
        q,
        f,
        perturbation_s_size=perturbation_s,
        perturbation_theta_size=perturbation_theta,
        savefig="results/figures/{}".format(save),
        true_beta=true_beta,
        derivatives_to_plot=derivatives_to_plot,
    )


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
