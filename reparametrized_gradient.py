import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bernoulli, norm, gaussian_kde
import tqdm

from utils import compute_score_bounds, convert_to_unit_vector

def expected_gradient_s_theta(agent_dist, theta, s, sigma):
    """Method that computes expected gradient ds/dtheta

    Keyword args:
    agent_dist -- AgentDistribution
    theta -- model parameters (D-1, 1) array
    s -- threshold (float)
    sigma -- standard deviation of the noise

    Returns:
    ds_dtheta - expected gradient
    """
    beta = convert_to_unit_vector(theta)
    r = s
    density = agent_dist.best_response_pdf(beta, s, sigma, r)
    pi_theta = expected_gradient_pi_theta(agent_dist, theta, s, sigma , r)
    pi_s = expected_gradient_pi_s(agent_dist, theta, s, sigma, r)
    val = -(1/(pi_s + density) ) * pi_theta
    return val

def empirical_gradient_s_theta(agent_dist, theta, s, sigma):
    """
    Empirical gradient ds/dtheta

    Keyword args:
    agent_dist -- AgentDistribution
    theta -- model parameters (D-1, 1) array
    s -- threshold (float)
    sigma -- standard deviation of the noise (float)

    Returns:
    ds_dtheta -- gradient estimate
    """
    beta = convert_to_unit_vector(theta)
    r = s

    hat_pi_theta = empirical_gradient_pi_theta(agent_dist, theta, s, sigma, r)
    hat_pi_s = empirical_gradient_pi_s(agent_dist, beta, s, sigma, r)
    hat_density = empirical_density(agent_dist, beta, s, sigma, r)

    val = -(1/(hat_pi_s + hat_density)) * hat_pi_theta
    return val

def empirical_density(agent_dist, theta, s, sigma, r, perturbation_size=0.1):

    """
    Empirical density p_beta,s(r).

    Keyword args:
    agent_dist -- AgentDistribution
    theta -- model parameters (D-1, 1) array
    s -- threshold (float)
    sigma -- standard deviation of the noise (float)


    Returns:
    density_estimate -- density estimate p_beta,s(r)
    """
    perturbations = (2 * bernoulli.rvs(p=0.5, size=agent_dist.n * (agent_dist.d-1)).reshape(agent_dist.n, agent_dist.d-1, 1) -1 ) * perturbation_size
    scores = []

    beta = convert_to_unit_vector(theta)
    bounds = compute_score_bounds(beta)

    interpolators_theta = []
    for agent in agent_dist.agents:
        f, _ = agent.br_score_function_beta(s, sigma)
        interpolators_theta.append(f)

    theta = np.arctan2(beta[1], beta[0])
    for i in range(int(agent_dist.n/2)):
        theta_perturbed = theta + perturbations[i]
        if theta_perturbed < -np.pi:
            theta_perturbed += 2 * np.pi
        if theta_perturbed > np.pi:
            theta_perturbed -= 2 * np.pi
        agent_type = agent_dist.n_agent_types[i]
        br_score = interpolators_theta[agent_type](theta_perturbed)
        scores.append(br_score.item())

    interpolators_s = []
    for agent in agent_dist.agents:
        interpolators_s.append(agent.br_score_function_s(beta, sigma))
    
    for i in range(int(agent_dist.n/2), agent_dist.n):
        s_perturbed = np.clip(s + perturbations[i], a_min=bounds[0], a_max=bounds[1])
        agent_type = agent_dist.n_agent_types[i]
        br_score = interpolators_s[agent_type](s_perturbed)
        scores.append(br_score.item())


    scores = np.array(scores)
    noise = norm.rvs(loc=0., scale=sigma, size=agent_dist.n)
    noisy_scores = np.clip(scores + noise, a_min=bounds[0], a_max=bounds[1])
    kde = gaussian_kde(noisy_scores)
    return kde(r).item()

def expected_gradient_pi_theta(agent_dist, theta, s, sigma, r):
    dim = agent_dist.d
    assert dim==2, "Method does not work for dimension {}".format(dim)

    beta = convert_to_unit_vector(theta)
    
    br_dist, grad_theta_dist = agent_dist.br_gradient_theta_distribution(theta, s, sigma)

    z = r - np.array([np.matmul(beta.T, x) for x in  br_dist]).reshape(len(br_dist), 1)
    prob = norm.pdf(z, loc=0., scale=sigma)
    dbeta_dtheta = np.array([-np.sin(theta), np.cos(theta)]).reshape(2, 1)
    first_term = np.array([np.matmul(grad_theta_dist[i].T, beta).item() for i in range(len(grad_theta_dist))]).reshape(agent_dist.n_types, 1)
    second_term = np.array([np.matmul(br_dist[i].T, dbeta_dtheta).item() for i in range(len(grad_theta_dist))]).reshape(agent_dist.n_types, 1)
    total = first_term + second_term
    res = - prob * total * agent_dist.prop.reshape(len(agent_dist.prop), 1)
    grad_pi_theta = np.sum(res)
    return grad_pi_theta.item()

def empirical_gradient_pi_theta(agent_dist, theta, s, sigma, r, perturbation_size=0.1):
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

    beta = convert_to_unit_vector(theta)
    bounds = compute_score_bounds(beta)
     
    interpolators = []
    for agent in agent_dist.agents:
        f, _ = agent.br_score_function_beta(s, sigma)
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
    return gamma_pi_theta.item()


def expected_gradient_loss_theta(agent_dist, theta, s, sigma, true_beta=None):
    """Method computes partial L(theta)/partial theta.

    Keyword args:
    agent_dist -- AgentDistribution
    theta -- model parameters
    s -- threshold
    sigma -- standard deviation of noise distribution
    true_beta -- optional ideal model


    Returns:
    d_l_d_beta -- expected gradient wrt to beta of policy loss at beta

    """
    dim = agent_dist.d
    assert dim==2, "Method does not work for dimension {}".format(dim)

    beta = convert_to_unit_vector(theta)
    if true_beta is None:
        true_beta = np.zeros((agent_dist.d, 1))
        true_beta[0] = 1.
    bounds = compute_score_bounds(beta)
    s = np.clip(s, a_min=bounds[0], a_max = bounds[1])
    true_scores = np.array([-np.matmul(true_beta.T, agent.eta).item() for agent in agent_dist.agents]).reshape(len(agent_dist.agents), 1)
    br_dist, jacobian_dist = agent_dist.br_gradient_beta_distribution(beta, s, sigma)
    z = s - np.array([np.matmul(beta.T, x) for x in  br_dist]).reshape(len(br_dist), 1)
    prob = norm.pdf(z, loc=0., scale=sigma)
    dbeta_dtheta = np.array([-np.sin(theta), np.cos(theta)]).reshape(2, 1)
    vec = np.array([np.matmul(beta.T, jacobian_dist[i]) + br_dist[i].T for i in range(len(br_dist))]).reshape(len(agent_dist.agents), 1, len(beta))
    final_scalar = np.array([np.matmul(v, dbeta_dtheta ).item() for v in vec]).reshape(len(agent_dist.agents), 1)
    res = prob * final_scalar * true_scores * agent_dist.prop.reshape(len(agent_dist.prop), 1)
    d_l_d_theta = np.sum(res, axis=0)
    return d_l_d_theta.item()

def empirical_gradient_loss_theta(agent_dist, theta, s, sigma, q, true_beta=None, perturbation_size=0.1):
    """Method that returns the empirical gradient of loss wrt to theta incurred given an agent distribution and model and threshold.
    Assumes that there is an model true_beta when applied to the agents' hidden eta features
    optimally selects the top agents.

    Keyword args:
    agent_dist -- AgentDistribution
    theta -- model parameters (D-1, 1) array
    s -- threshold (float)
    q -- quantile
    sigma -- standard deviation of the noise (float)
    true_beta -- (D, 1) array

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
    return gamma_loss_theta.item()



def plot_grad_loss_theta(agent_dist, sigma, q, f, true_beta=None, savefig=None):
    grad_theta = []
    emp_grad_theta = []
    thetas = np.linspace(-np.pi, np.pi, 50)
    for theta in tqdm.tqdm(thetas):
        s_beta = f(theta)
        grad = expected_gradient_loss_theta(agent_dist, theta, s_beta, sigma, true_beta)
        grad_theta.append(grad.item())
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
