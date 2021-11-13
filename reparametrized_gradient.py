import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bernoulli, norm
import tqdm

from utils import compute_score_bounds

def expected_gradient_pi_theta(agent_dist, theta, sigma, r, f):
    dim = agent_dist.d
    assert dim==2, "Method does not work for dimension {}".format(dim)

    beta = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)
    s = f(theta)
    
    br_dist, jacobian_dist = agent_dist.br_gradient_beta_distribution(beta, s, sigma)

    z = r - np.array([np.matmul(beta.T, x) for x in  br_dist]).reshape(len(br_dist), 1)
    prob = norm.pdf(z, loc=0., scale=sigma)
    vec = np.array([np.matmul(beta.T, jacobian_dist[i]) + br_dist[i].T for i in range(len(br_dist))]).reshape(len(agent_dist.agents), len(beta))
    dbeta_dtheta = np.array([-np.sin(theta), np.cos(theta)]).reshape(2, 1)
    vec_chain = np.array([np.matmul(x.T, dbeta_dtheta).item() for x in vec]).reshape(2, 1)
    res = -prob * vec_chain * agent_dist.prop.reshape(len(agent_dist.prop), 1)
    grad_pi_theta = np.sum(res, axis=0)

    return grad_pi_theta

def empirical_gradient_pi_theta(agent_dist, beta, s, sigma, r, perturbation_size=0.1):
    """Method that returns the empirical gradient of pi wrt to beta incurred given an agent distribution and model and threshold.
    Assumes that there is an model true_beta when applied to the agents' hidden eta features
    optimally selects the top agents.

    Keyword args:
    agent_dist -- AgentDistribution
    beta -- model parameters (N, 1) array
    s -- threshold (float)
    sigma -- standard deviation of the noise (float)

    Returns:
    gamma_pi_beta - gradient of pi at beta
    """
    perturbations = (2 * bernoulli.rvs(p=0.5, size=agent_dist.n * (agent_dist.d-1)).reshape(agent_dist.n, agent_dist.d-1, 1) -1 ) * perturbation_size
    scores = []
    
    bounds = compute_score_bounds(beta)
     
    interpolators = []
    for agent in agent_dist.agents:
        f = agent.br_score_function_beta(s, sigma)
        interpolators.append(f)

    for i in range(agent_dist.n):
        theta_perturbed = theta + perturbations[i]
        if theta_perturbed < -np.pi:
            theta_perturbed += 2 * np.pi
        if theta_perturbed > np.pi:
            theta_perturbed -= 2 * np.pi
        agent_type = agent_dist.n_agent_types[i]
        br_score = interpolators[agent_type](theta_perturbed)
        scores.append(br_score.item())

    scores = np.array(scores).reshape(agent_dist.n, 1)
    noise = norm.rvs(loc=0., scale=sigma, size=agent_dist.n).reshape(agent_dist.n, 1)
    noisy_scores = np.clip(scores + noise, a_min=bounds[0], a_max=bounds[1])
    indicators = noisy_scores <= r
    
    perturbations = perturbations.reshape(agent_dist.n, agent_dist.d-1)
    Q = np.matmul(perturbations.T, perturbations)
    gamma_pi_theta = np.linalg.solve(Q, np.matmul(perturbations.T, indicators))
    return gamma_pi_theta


def expected_gradient_loss_theta(agent_dist, theta, sigma, f, true_beta=None):
    """Method computes partial L(theta)/partial theta.

    Keyword args:
    agent_dist -- AgentDistribution
    beta -- model parameters
    sigma -- standard deviation of noise distribution
    f -- function that maps arctan(beta[1]/beta[0]) -> s_beta (fixed point)
    true_beta -- optional ideal model


    Returns:
    d_l_d_beta -- expected gradient wrt to beta of policy loss at beta

    """
    dim = agent_dist.d
    assert dim==2, "Method does not work for dimension {}".format(dim)

    beta = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)
    if true_beta is None:
        true_beta = np.zeros((agent_dist.d, 1))
        true_beta[0] = 1.
    bounds = compute_score_bounds(beta)
    s = np.clip(f(theta), a_min=bounds[0], a_max = bounds[1])
    true_scores = np.array([-np.matmul(true_beta.T, agent.eta).item() for agent in agent_dist.agents]).reshape(len(agent_dist.agents), 1)
    br_dist, jacobian_dist = agent_dist.br_gradient_beta_distribution(beta, s, sigma)
    z = s - np.array([np.matmul(beta.T, x) for x in  br_dist]).reshape(len(br_dist), 1)
    prob = norm.pdf(z, loc=0., scale=sigma)
    dbeta_dtheta = np.array([-np.sin(theta), np.cos(theta)]).reshape(2, 1)
    vec = np.array([np.matmul(beta.T, jacobian_dist[i]) + br_dist[i].T for i in range(len(br_dist))]).reshape(len(agent_dist.agents), 1, len(beta))
    final_scalar = np.array([np.matmul(v, dbeta_dtheta ).item() for v in vec]).reshape(len(agent_dist.agents), 1)
    res = prob * final_scalar * true_scores * agent_dist.prop.reshape(len(agent_dist.prop), 1)
    d_l_d_theta = np.sum(res, axis=0)
    return d_l_d_theta

def empirical_gradient_loss_theta(agent_dist, theta, s, sigma, q, true_beta=None, perturbation_size=0.1):
    """Method that returns the empirical gradient of loss wrt to theta incurred given an agent distribution and model and threshold.
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
    beta = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)
    if true_beta is None:
        true_beta = np.zeros(beta.shape)
        true_beta[0] = 1.
        
    perturbations = (2 * bernoulli.rvs(p=0.5, size=agent_dist.n * (agent_dist.d-1)).reshape(agent_dist.n, agent_dist.d-1, 1) -1 ) * perturbation_size
    scores = []
    

    bounds = compute_score_bounds(beta)
    
    interpolators = []
    for agent in agent_dist.agents:
        f = agent.br_score_function_beta(s, sigma)
        interpolators.append(f)
    
    for i in range(agent_dist.n):
        theta_perturbed = theta + perturbations[i]
        if theta_perturbed < -np.pi:
            theta_perturbed += 2 * np.pi
        if theta_perturbed > np.pi:
            theta_perturbed -= 2 * np.pi
        agent_type = agent_dist.n_agent_types[i]
        br_score = interpolators[agent_type](theta_perturbed)
        scores.append(br_score.item())
        
    scores = np.array(scores).reshape(agent_dist.n, 1)
    noise = norm.rvs(loc=0., scale=sigma, size=agent_dist.n).reshape(agent_dist.n, 1)
    noisy_scores = np.clip(scores + noise, a_min=bounds[0], a_max=bounds[1])
    
    #Compute loss
    treatments = noisy_scores >= np.quantile(noisy_scores, q)
    loss_vector = treatments * np.array([-np.matmul(true_beta.T, eta).item() for eta in agent_dist.get_etas()]).reshape(agent_dist.n, 1)
    
    perturbations = perturbations.reshape(agent_dist.n, agent_dist.d-1)
    Q = np.matmul(perturbations.T, perturbations)
    gamma_loss_theta = np.linalg.solve(Q, np.matmul(perturbations.T, loss_vector))
    return gamma_loss_theta



def plot_grad_loss_theta(agent_dist, sigma, q, f, true_beta=None, savefig=None):
    grad_theta = []
    emp_grad_theta = []
    thetas = np.linspace(-np.pi, np.pi, 50)
    for theta in tqdm.tqdm(thetas):
        grad = expected_gradient_loss_theta(agent_dist, theta, sigma, f, true_beta)
        grad_theta.append(grad.item())
        
        s_beta = f(theta)
        emp_grad = empirical_gradient_loss_theta(agent_dist, theta, s_beta, sigma, q, true_beta)
        emp_grad_theta.append(emp_grad.item())


    plt.plot(thetas, emp_grad_theta, label="empirical")
    plt.plot(thetas, grad_theta, label="expected")
    plt.xlabel("Theta (Corresponds to Beta)")
    plt.ylabel("dL/dtheta")
    plt.title("Theta vs. dL/dtheta")
    plt.legend()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close()
