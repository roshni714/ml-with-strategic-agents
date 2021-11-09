import numpy as np
from scipy.stats import norm, bernoulli
from utils import compute_score_bounds
import tqdm
import matplotlib.pyplot as plt

def expected_gradient_loss_beta(agent_dist, theta, sigma, f, true_beta=None):
    """Method computes partial L(beta)/partial beta.

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
        true_beta = np.zeros(beta.shape)
        true_beta[0] = 1.
    bounds = compute_score_bounds(beta)
    s = np.clip(f(theta), a_min=bounds[0], a_max = bounds[1])
    true_scores = np.array([-np.matmul(true_beta.T, agent.eta).item() for agent in agent_dist.agents]).reshape(len(agent_dist.agents), 1)
    br_dist, jacobian_dist = agent_dist.br_gradient_beta_distribution(beta, s, sigma)
    z = s - np.array([np.matmul(beta.T, x) for x in  br_dist]).reshape(len(br_dist), 1)
    prob = norm.pdf(z, loc=0., scale=sigma)
    vec = np.array([np.matmul(beta.T, jacobian_dist[i]) + br_dist[i].T for i in range(len(br_dist))]).reshape(len(agent_dist.agents), len(beta))
    res = prob * vec * true_scores * agent_dist.prop.reshape(len(agent_dist.prop), 1)

    
    d_l_d_beta = np.sum(res, axis=0)
    return d_l_d_beta

def empirical_gradient_loss_beta(agent_dist, beta, s, sigma, q, true_beta=None, perturbation_size=0.1):
    """Method that returns the empirical gradient of loss wrt to beta incurred given an agent distribution and model and threshold.
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
        
    perturbations = (2 * bernoulli.rvs(p=0.5, size=agent_dist.n * agent_dist.d).reshape(agent_dist.n, agent_dist.d, 1) -1 ) * perturbation_size
    scores = []
    
    bounds = compute_score_bounds(beta)
    
    for i in range(agent_dist.n):
        beta_perturbed = beta + perturbations[i]
        agent = agent_dist.agents[agent_dist.n_agent_types[i]]
        br = agent.best_response(beta_perturbed, s, sigma)
        scores.append(np.matmul(beta_perturbed.T, br).item())
        
    scores = np.array(scores).reshape(agent_dist.n, 1)
    noise = norm.rvs(loc=0., scale=sigma, size=agent_dist.n).reshape(agent_dist.n, 1)
    noisy_scores = np.clip(scores + noise, a_min=bounds[0], a_max=bounds[1])
    
    #Compute loss
    treatments = noisy_scores >= np.quantile(noisy_scores, q)
    loss_vector = treatments * np.array([-np.matmul(true_beta.T, eta).item() for eta in agent_dist.get_etas()]).reshape(agent_dist.n, 1)
    
    perturbations = perturbations.reshape(agent_dist.n, agent_dist.d)
    Q = np.matmul(perturbations.T, perturbations)
    gamma_loss_beta = np.linalg.solve(Q, np.matmul(perturbations.T, loss_vector))
    return gamma_loss_beta

def expected_gradient_loss_s(agent_dist, theta, sigma, f, true_beta=None):
    """Method computes partial L(beta)/partial s.

    Keyword args:
    agent_dist -- AgentDistribution
    beta -- model parameters
    sigma -- standard deviation of noise distribution
    f -- function that maps arctan(beta[1]/beta[0]) -> s_beta (fixed point)
    true_beta -- optional ideal model


    Returns:
    d_l_d_s -- expected gradient wrt to s of policy loss 

    """
    dim = agent_dist.d
    assert dim==2, "Method does not work for dimension {}".format(dim)

    beta = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)
    if true_beta is None:
        true_beta = np.zeros(beta.shape)
        true_beta[0] = 1.
    bounds = compute_score_bounds(beta)
    s = np.clip(f(theta), a_min=bounds[0], a_max = bounds[1])
    true_scores = np.array([-np.matmul(true_beta.T, agent.eta).item() for agent in agent_dist.agents]).reshape(len(agent_dist.agents), 1)
    br_dist, grad_s_dist  = agent_dist.br_gradient_s_distribution(beta, s, sigma)
    z = s - br_dist
    prob = norm.pdf(z, loc=0., scale=sigma)
    vec = np.array([1 - np.matmul(beta.T, grad_s_dist[i]) for i in range(len(br_dist))])
    res = vec * true_scores * agent_dist.prop.reshape(len(agent_dist.prop), 1)

    d_l_d_s = np.mean(res)
    return d_l_d_s

def plot_grad_loss_beta(agent_dist, sigma, q, f, true_beta=None, savefig=None):
    grad_beta1 = []
    grad_beta2 = []
    emp_grad_beta1 = []
    emp_grad_beta2 = []
    thetas = np.linspace(-np.pi, np.pi, 50)
    for theta in tqdm.tqdm(thetas):
        vec_beta = expected_gradient_loss_beta(agent_dist, theta, sigma, f, true_beta)
        grad_beta1.append(vec_beta[0].item())
        grad_beta2.append(vec_beta[1].item())
        
        s_beta = f(theta)
        beta = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)
        emp_vec_beta = empirical_gradient_loss_beta(agent_dist, beta, s_beta, sigma, q, true_beta)
        emp_grad_beta1.append(emp_vec_beta[0].item())
        emp_grad_beta2.append(emp_vec_beta[1].item())

    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    ax[0].plot(thetas, emp_grad_beta1, label="empirical")
    ax[0].plot(thetas, grad_beta1, label="expected")
    ax[0].set_xlabel("Theta (Corresponds to Beta)")
    ax[0].set_ylabel("dL/dbeta1")
    ax[0].set_title("Beta vs. dL/dbeta1")
    ax[1].plot(thetas, emp_grad_beta2, label="empirical")
    ax[1].plot(thetas, grad_beta2, label="expected")
    ax[1].set_xlabel("Theta (Corresponds to Beta)")
    ax[1].set_ylabel("dL/dbeta2")
    ax[1].set_title("Beta vs. dL/dbeta2")
    plt.legend()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close()
    
    return thetas, grad_beta1, grad_beta2, emp_grad_beta1, emp_grad_beta2

def plot_grad_loss_s(agent_dist, sigma, f, true_beta=None, savefig=None):
    grad_s = []
    thetas = np.linspace(-np.pi, np.pi, 50)
    for theta in thetas:
        d_l_d_s = expected_gradient_loss_s(agent_dist, theta, sigma, f, true_beta=None)
        grad_s.append(d_l_d_s)
    plt.plot(thetas, grad_s, label="expected")
    plt.plot(thetas, emp_grad_s, label="empirical")
    plt.xlabel("Theta (Corresponds to Beta)")
    plt.ylabel("dL/s")
    plt.title("Beta vs. dL/ds")
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close()
    
    return thetas, grad_s


