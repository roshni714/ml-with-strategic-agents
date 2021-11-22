import numpy as np
import argh

from agent_distribution import AgentDistribution
from gradient_estimation import GradientEstimator
from expected_gradient import ExpectedGradient
from reporting import report_results
from utils import compute_continuity_noise, fixed_point_interpolation_true_distribution
from optimal_beta import optimal_beta_expected_policy_loss


def learn_model(
    agent_dist,
    sigma,
    q,
    f,
    true_beta=None,
    learning_rate=0.05,
    max_iter=30,
    perturbation_s=0.1,
    perturbation_theta=0.1,
):
    if true_beta is None:
        true_beta = np.zeros((agent_dist.d, 1))
        true_beta[0] = 1.0

    #    theta = np.arctan2(true_beta[1], true_beta[0]).item()
    theta = np.random.uniform(-np.pi, np.pi, 1).item()
    thetas = []
    emp_losses = []
    for i in range(max_iter):
        # Get equilibrium cutoff
        if theta < -np.pi:
            theta += 2 * np.pi
        if theta > np.pi:
            theta -= 2 * np.pi

        s_eq = f(theta)
        thetas.append(theta)
        grad_est = ExpectedGradient(
            agent_dist,
            theta,
            s_eq,
            sigma,
#            q,
            true_beta,
#            perturbation_s_size=perturbation_s,
#            perturbation_theta_size=perturbation_theta,
        )
        dic = grad_est.compute_total_derivative()
        loss = dic["loss"]
        grad_theta = dic["total_deriv"]
        emp_losses.append(loss)
        print(
            "Loss: {}".format(loss),
            "Theta:{}".format(theta),
            "Gradient: {}".format(grad_theta),
        )
        theta -= grad_theta * learning_rate

    return thetas, emp_losses


@argh.arg("--n", default=100000)
@argh.arg("--perturbation_s", default=0.1)
@argh.arg("--perturbation_theta", default=0.1)
@argh.arg("--learning_rate", default=1.0)
@argh.arg("--max_iter", default=500)
@argh.arg("--save", default="results")
def main(
    n=100000,
    perturbation_s=0.1,
    perturbation_theta=0.1,
    learning_rate=1.0,
    max_iter=500,
    save="results",
):
    np.random.seed(0)

    n_types = 20
    d = 2
    etas = np.random.uniform(0.3, 0.8, n_types * d).reshape(n_types, d, 1)
    gammas = np.random.uniform(5.0, 8.0, n_types * d).reshape(n_types, d, 1)
    dic = {"etas": etas, "gammas": gammas}
    agent_dist = AgentDistribution(n=n, d=d, n_types=n_types, types=dic, prop=None)
    #    sigma = compute_continuity_noise(agent_dist)
    sigma = 0.35
    q = 0.7
    f = fixed_point_interpolation_true_distribution(
        agent_dist, sigma, q, plot=False, savefig=None
    )
    thetas, emp_losses = learn_model(
        agent_dist,
        sigma,
        q,
        f,
        true_beta=None,
        learning_rate=learning_rate,
        max_iter=max_iter,
        perturbation_s=perturbation_s,
        perturbation_theta=perturbation_theta,
    )

    min_loss, opt_beta, _, _, _ = optimal_beta_expected_policy_loss(
        agent_dist, sigma, f, plot=False
    )
    opt_theta = np.arctan2(opt_beta[1], opt_beta[0]).item()

    results = {
        "n": n,
        "d": d,
        "n_types": n_types,
        "sigma": sigma,
        "q": q,
        "opt_loss": min_loss,
        "opt_theta": opt_theta,
        "perturbation_s": perturbation_s,
        "perturbation_theta": perturbation_theta,
        "final_theta": thetas[-1],
        "final_loss": emp_losses[-1],
    }
    assert len(thetas) == len(emp_losses)
    print(results)
    report_results(results, thetas, emp_losses, save)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
