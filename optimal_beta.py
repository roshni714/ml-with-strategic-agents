import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import interp1d

from agent_distribution import AgentDistribution
from utils import compute_continuity_noise, compute_contraction_noise, compute_score_bounds

def fixed_point_interpolation_true_distribution(agent_dist, sigma, q, plot=False, savefig=None):
    """Method that returns a function that maps model parameters to the fixed point it induces.

    The function is estimated by doing a linear interpolation of the fixed points from theta
    (a 1-dimensional parametrization of beta). theta -> beta = [cos (theta),  sin(theta)]
    The function maps theta -> s_beta.

    Keyword args:
    agent_dist -- AgentDistribution
    sigma -- standard deviation of the noise distribution (float)
    q -- quantile (float)
    plot -- optional plotting argument
    savefig -- path to save figure

    Returns:
    f -- interp1d object that maps theta to s_beta
    """
    dim = agent_dist.d
    assert dim==2, "Method does not work for dimension {}".format(dim)

    thetas = np.linspace(-np.pi, np.pi, 50)
    fixed_points = []
    betas = []

    #compute beta and fixed point for each theta
    print("Computing fixed points...")
    for theta in thetas:
        beta = np.array([np.cos(theta), np.sin(theta)]).reshape(dim, 1)
        fp = agent_dist.quantile_fixed_point_true_distribution(beta, sigma, q, plot=True)
        fixed_points.append(fp)
        betas.append(beta)

    f = interp1d(thetas, fixed_points, kind="cubic")

    if plot:
        plt.plot(thetas, fixed_points, label="actual")
        plt.plot(thetas, f(thetas), label="interpolation")
        plt.xlabel("Thetas (corresponds to different Beta)")
        plt.ylabel("s_beta")
        plt.title("Location of Fixed Points: s_beta vs. beta")
        plt.legend()
        if savefig is not None:
            plt.savefig(savefig)
        plt.show()
        plt.close()

    return f

def empirical_policy_loss(agent_dist, beta, s, sigma, q, true_beta=None):
    """Method that returns the empirical policy loss incurred given an agent distribution and model and threshold.
    Assumes that there is an model true_beta when applied to the agents' hidden eta features
    optimally selects the top agents.

    Keyword args:
    agent_dist -- AgentDistribution
    beta -- model parameters (N, 1) array
    s -- threshold (float)
    q -- quantile
    sigma -- standard deviation of the noise (float)
    true_beta -- (N, 1) array

    Returns:
    loss -- empirical policy loss
    """
    if true_beta is None:
        true_beta = np.zeros(beta.shape)
        true_beta[0] = 1.

    true_agent_types = agent_dist.n_agent_types
    etas = agent_dist.get_etas()
    true_scores = np.array([np.matmul(true_beta.T, eta).item() for eta in etas]).reshape(agent_dist.n, 1)

    br_dist = agent_dist.best_response_score_distribution(beta, s, sigma)
    n_br = br_dist[agent_dist.n_agent_types]
    curr_bounds = compute_score_bounds(beta)

    noisy_scores = norm.rvs(loc=0.0, scale=sigma, size=agent_dist.n)
    noisy_scores += n_br
    noisy_scores = np.clip(noisy_scores, a_min=curr_bounds[0], a_max=curr_bounds[1]).reshape(agent_dist.n, 1)
    x = np.quantile(noisy_scores, q)
    loss = -np.mean(true_scores * (noisy_scores >= x))
    return loss

def optimal_beta_empirical_policy_loss(agent_dist, sigma, q, f, true_beta=None, plot=False, savefig=None):
    """Method returns the model parameters that minimize the empirical policy loss.

    Keyword args:
    agent_dist -- AgentDistribution
    sigma -- standard deviation of noise distribution
    q -- quantile
    f -- function that maps arctan(beta[1]/beta[0]) -> s_beta (fixed point)
    true_beta -- optional ideal model
    plot -- optional plotting
    savefig -- path to save figure

    Returns:
    min_loss -- minimum loss (float)
    opt_beta -- beta that minimizes the loss (2, 1) array
    opt_s_beta -- optimal threshold (float)
    """
    dim = agent_dist.d
    assert dim==2, "Method does not work for dimension {}".format(dim)

    thetas = np.linspace(-np.pi, np.pi, 50)
    losses = []
    for theta in thetas:
        beta = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)
        loss = empirical_policy_loss(agent_dist, beta, f(theta), sigma, q, true_beta=true_beta)
        losses.append(loss)

    idx = np.argmin(losses)
    min_loss = losses[idx]
    opt_beta = np.array([np.cos(thetas[idx]), np.sin(thetas[idx])]).reshape(2, 1)
    opt_s_beta = f(thetas[idx])
    if plot:
        plt.plot(thetas, losses)
        plt.xlabel("Theta (Represents Beta)")
        plt.ylabel("Empirical Loss")
        plt.title("Empirical Loss Incurred at Different Beta")
        if savefig:
            plt.savefig(savefig)

    return min_loss, opt_beta, opt_s_beta

