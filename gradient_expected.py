import numpy as np
from scipy.stats import norm, bernoulli
from utils import compute_score_bounds, convert_to_unit_vector
import tqdm
import matplotlib.pyplot as plt


def empirical_gradient_pi_s(agent_dist, theta, s, sigma, r, perturbation_size=0.05):
    """Method that returns the empirical gradient of pi wrt to s incurred given an agent distribution and model and threshold.

    Keyword args:
    agent_dist -- AgentDistribution
    theta -- model parameters (D-1, 1) array
    s -- threshold (float)
    sigma -- standard deviation of noise distribution (float)
    r -- value that CDF should be evaluated at

    Returns:
    gamma_loss_s -- empirical gradient dL/ds
    """
    beta = convert_to_unit_vector(theta)  
    perturbations = (2 * bernoulli.rvs(p=0.5, size=agent_dist.n).reshape(agent_dist.n, 1) -1 ) * perturbation_size
    scores = []
    
    bounds = compute_score_bounds(beta)
    interpolators = []
    
    for agent in agent_dist.agents:
        interpolators.append(agent.br_score_function_s(beta, sigma))
    
    for i in range(agent_dist.n):
        s_perturbed = np.clip(s + perturbations[i], a_min=bounds[0], a_max=bounds[1])
        agent_type = agent_dist.n_agent_types[i]
        br_score = interpolators[agent_type](s_perturbed)
        scores.append(br_score.item())
        
    scores = np.array(scores).reshape(agent_dist.n, 1)
    noise = norm.rvs(loc=0., scale=sigma, size=agent_dist.n).reshape(agent_dist.n, 1)
    noisy_scores =  np.clip(scores + noise, a_min = bounds[0], a_max = bounds[1])
    
    indicators = noisy_scores <= r
    
    Q = np.matmul(perturbations.T, perturbations)
    gamma_pi_s = np.linalg.solve(Q, np.matmul(perturbations.T, indicators))
    return gamma_pi_s.item()

def expected_gradient_pi_s(agent_dist, theta, s, sigma, r):
    """Method computes partial pi/partial s.

    Keyword args:
    agent_dist -- AgentDistribution
    theta -- model parameters
    sigma -- standard deviation of noise distribution
    f -- function that maps arctan(beta[1]/beta[0]) -> s_beta (fixed point)

    Returns:
    d_pi_d_s -- expected gradient wrt to s of policy loss 

    """
    dim = agent_dist.d
    assert dim==2, "Method does not work for dimension {}".format(dim)

    beta = convert_to_unit_vector(theta)
    bounds = compute_score_bounds(beta)
    br_dist, grad_s_dist  = agent_dist.br_gradient_s_distribution(beta, s, sigma)
    
    z = r - np.array([np.matmul(beta.T, x) for x in  br_dist]).reshape(len(br_dist), 1)
    
    prob = norm.pdf(z, loc=0., scale=sigma)
    vec = np.array([np.matmul(beta.T, grad_s_dist[i]).item() for i in range(len(br_dist))]).reshape(agent_dist.n_types, 1)
    res = -prob * vec * agent_dist.prop.reshape(agent_dist.n_types, 1)

    d_l_d_s = np.sum(res)
    return d_l_d_s.item()

def plot_grad_pi_s(agent_dist, sigma, f, savefig=None):
    emp_grad_s = []
    grad_s = []
    sigma = 0.35
    theta = np.pi/4
    beta = convert_to_unit_vector(theta)
    s_beta = f(theta)
    bounds = compute_score_bounds(beta)
    rs = np.linspace(bounds[0], bounds[1], 50)
    for r in rs:
        emp_grad = empirical_gradient_pi_s(agent_dist, beta, s_beta, sigma, r)
        grad = expected_gradient_pi_s(agent_dist, theta, s_beta, sigma, r)
        emp_grad_s.append(emp_grad)
        grad_s.append(grad)
    plt.plot(rs, emp_grad_s, label="empirical")
    plt.plot(rs, grad_s, label="expected")
    plt.legend()
    plt.xlabel("r")
    plt.ylabel("dpi(r)/ds")
    plt.title("r vs. dpi(r)/ds")
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close()

def plot_grad_pi_beta(agent_dist, sigma, f, savefig=None):
    grad_betas_0 = []
    grad_betas_1 = []
    emp_grad_betas_0 = []
    emp_grad_betas_1 = []
    theta = np.pi/4
    beta = convert_to_unit_vector(theta)
    sigma = 0.35
    s_beta = f(theta)
    bounds = compute_score_bounds(beta)
    rs = np.linspace(bounds[0], bounds[1], 50)
    for r in rs:
        grad_beta = expected_gradient_pi_beta(agent_dist, theta, s_beta, sigma, r)
        grad_betas_0.append(grad_beta[0])
        grad_betas_1.append(grad_beta[1])
        emp_grad_beta = empirical_gradient_pi_beta(agent_dist, theta, s_beta, sigma, r)
        emp_grad_betas_0.append(emp_grad_beta[0])
        emp_grad_betas_1.append(emp_grad_beta[1])
    
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].plot(rs, emp_grad_betas_0, label="empirical")
    ax[0].plot(rs, grad_betas_0, label="expected")
    ax[0].set_xlabel("r")
    ax[0].set_ylabel("dpi(r)/dbeta1")
    ax[0].set_title("r vs. dpi(r)/dbeta1")
    ax[1].plot(rs, emp_grad_betas_1, label="empirical")
    ax[1].plot(rs, grad_betas_1, label="expected")
    ax[1].set_xlabel("r")
    ax[1].set_ylabel("dpi(r)/dbeta2")
    ax[1].set_title("r vs. dpi(r)/dbeta2")
    plt.legend()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close()

def expected_gradient_pi_beta(agent_dist, theta, s, sigma, r):
    dim = agent_dist.d
    assert dim==2, "Method does not work for dimension {}".format(dim)

    beta = convert_to_unit_vector(theta) 
    br_dist, jacobian_dist = agent_dist.br_gradient_beta_distribution(beta, s, sigma)

    z = r - np.array([np.matmul(beta.T, x) for x in  br_dist]).reshape(len(br_dist), 1)
    prob = norm.pdf(z, loc=0., scale=sigma)
    vec = np.array([np.matmul(beta.T, jacobian_dist[i]) + br_dist[i].T for i in range(len(br_dist))]).reshape(len(agent_dist.agents), len(beta))
    res = -prob * vec * agent_dist.prop.reshape(len(agent_dist.prop), 1)
    grad_pi_beta = np.sum(res, axis=0)

    return grad_pi_beta

def empirical_gradient_pi_beta(agent_dist, theta, s, sigma, r, perturbation_size=0.1):
    """Method that returns the empirical gradient of pi wrt to beta incurred given an agent distribution and model and threshold.

    Keyword args:
    agent_dist -- AgentDistribution
    theta  -- model parameters (D-1, 1) array
    s -- threshold (float)
    sigma -- standard deviation of the noise (float)

    Returns:
    gamma_pi_beta - gradient of pi at beta
    """
    perturbations = (2 * bernoulli.rvs(p=0.5, size=agent_dist.n * agent_dist.d).reshape(agent_dist.n, agent_dist.d, 1) -1 ) * perturbation_size
    scores = []

    beta = convert_to_unit_vector(theta)
    bounds = compute_score_bounds(beta)
    
    for i in range(agent_dist.n):
        beta_perturbed = beta + perturbations[i]
        agent = agent_dist.agents[agent_dist.n_agent_types[i]]
        br = agent.best_response(beta, s, sigma)
        scores.append(np.matmul(beta_perturbed.T, br).item())
        
    scores = np.array(scores).reshape(agent_dist.n, 1)
    noise = norm.rvs(loc=0., scale=sigma, size=agent_dist.n).reshape(agent_dist.n, 1)
    noisy_scores = np.clip(scores + noise, a_min=bounds[0], a_max=bounds[1])
    
    indicators = noisy_scores <= r
    
    perturbations = perturbations.reshape(agent_dist.n, agent_dist.d)
    Q = np.matmul(perturbations.T, perturbations)
    gamma_pi_beta = np.linalg.solve(Q, np.matmul(perturbations.T, indicators))
    return gamma_pi_beta

def expected_gradient_loss_beta(agent_dist, theta, s, sigma, true_beta=None):
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

    beta = convert_to_unit_vector(theta)
    if true_beta is None:
        true_beta = np.zeros(beta.shape)
        true_beta[0] = 1.
    bounds = compute_score_bounds(beta)
    true_scores = np.array([-np.matmul(true_beta.T, agent.eta).item() for agent in agent_dist.agents]).reshape(len(agent_dist.agents), 1)
    br_dist, jacobian_dist = agent_dist.br_gradient_beta_distribution(beta, s, sigma)
    z = s - np.array([np.matmul(beta.T, x) for x in  br_dist]).reshape(len(br_dist), 1)
    prob = norm.pdf(z, loc=0., scale=sigma)
    vec = np.array([np.matmul(beta.T, jacobian_dist[i]) + br_dist[i].T for i in range(len(br_dist))]).reshape(len(agent_dist.agents), len(beta))
    res = prob * vec * true_scores * agent_dist.prop.reshape(len(agent_dist.prop), 1)

    
    d_l_d_beta = np.sum(res, axis=0)
    return d_l_d_beta

def empirical_gradient_loss_beta(agent_dist, theta, s, sigma, q, true_beta=None, perturbation_size=0.1):
    """Method that returns the empirical gradient of loss wrt to beta incurred given an agent distribution and model and threshold.
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
    if true_beta is None:
        true_beta = np.zeros(beta.shape)
        true_beta[0] = 1.
        
    perturbations = (2 * bernoulli.rvs(p=0.5, size=agent_dist.n * agent_dist.d).reshape(agent_dist.n, agent_dist.d, 1) -1 ) * perturbation_size
    scores = []
    
    beta = convert_to_unit_vector(theta)
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

def expected_gradient_loss_s(agent_dist, theta, s, sigma, true_beta=None):
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

    beta = convert_to_unit_vector(theta)
    if true_beta is None:
        true_beta = np.zeros(beta.shape)
        true_beta[0] = 1.
    bounds = compute_score_bounds(beta)
    s = np.clip(f(theta), a_min=bounds[0], a_max = bounds[1])
    true_scores = np.array([np.matmul(true_beta.T, agent.eta).item() for agent in agent_dist.agents]).reshape(len(agent_dist.agents), 1)
    br_dist, grad_s_dist  = agent_dist.br_gradient_s_distribution(beta, s, sigma)
    z = s - np.array([np.matmul(beta.T, x) for x in  br_dist]).reshape(len(br_dist), 1)
    prob = norm.pdf(z, loc=0., scale=sigma)
    vec = np.array([- np.matmul(beta.T, grad_s_dist[i]) for i in range(len(br_dist))]).reshape(agent_dist.n_types, 1)
    res = prob * vec * true_scores * agent_dist.prop.reshape(agent_dist.n_types, 1)

    d_l_d_s = np.sum(res)
    return d_l_d_s.item()

def empirical_gradient_loss_s(agent_dist, theta, s, sigma, q, true_beta=None, perturbation_size=0.05):
    """Method that returns the empirical gradient of loss wrt to s incurred given an agent distribution and model and threshold.
    Assumes that there is an model true_beta when applied to the agents' hidden eta features
    optimally selects the top agents.

    Keyword args:
    agent_dist -- AgentDistribution
    beta -- model parameters (N, 1) array
    s -- threshold (float)
    sigma -- standard deviation of noise distribution (float)
    q -- quantile
    true_beta -- (N, 1) array

    Returns:
    gamma_loss_s -- empirical gradient dL/ds
    """
    beta = convert_to_unit_vector(theta)
    if true_beta is None:
        true_beta = np.zeros(beta.shape)
        true_beta[0] = 1.
        
    perturbations = (2 * bernoulli.rvs(p=0.5, size=agent_dist.n).reshape(agent_dist.n, 1) -1 ) * perturbation_size
    scores = []
    
    bounds = compute_score_bounds(beta)
    interpolators = []
    
    for agent in agent_dist.agents:
        interpolators.append(agent.br_score_function_s(beta, sigma))
    
    for i in range(agent_dist.n):
        s_perturbed = np.clip(s + perturbations[i], a_min=bounds[0], a_max=bounds[1])
        agent_type = agent_dist.n_agent_types[i]
        br_score = interpolators[agent_type](s_perturbed)
        scores.append(br_score.item())
        
    scores = np.array(scores).reshape(agent_dist.n, 1)
    noise = norm.rvs(loc=0., scale=sigma, size=agent_dist.n).reshape(agent_dist.n, 1)
    noisy_scores = np.clip(scores + noise, a_min=bounds[0], a_max=bounds[1])
    
    perturbed_noisy_scores =  noisy_scores - perturbations
    
    #Compute loss
    treatments = perturbed_noisy_scores >= np.quantile(perturbed_noisy_scores, q)
    loss_vector = treatments * np.array([-np.matmul(true_beta.T, eta).item() for eta in agent_dist.get_etas()]).reshape(agent_dist.n, 1)
    
    Q = np.matmul(perturbations.T, perturbations)
    gamma_loss_s = np.linalg.solve(Q, np.matmul(perturbations.T, loss_vector))
    return gamma_loss_s.item()

def plot_grad_loss_beta(agent_dist, sigma, q, f, true_beta=None, savefig=None):
    grad_beta1 = []
    grad_beta2 = []
    emp_grad_beta1 = []
    emp_grad_beta2 = []
    thetas = np.linspace(-np.pi, np.pi, 50)
    for theta in tqdm.tqdm(thetas):
        s_beta = f(theta)
        vec_beta = expected_gradient_loss_beta(agent_dist, theta, s_beta, sigma, true_beta)
        grad_beta1.append(vec_beta[0].item())
        grad_beta2.append(vec_beta[1].item())
        
        emp_vec_beta = empirical_gradient_loss_beta(agent_dist, theta, s_beta, sigma, q, true_beta)
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

def plot_grad_loss_s(agent_dist, sigma, q, f, true_beta=None, savefig=None):
    grad_s = []
    emp_grad_s = []
    thetas = np.linspace(-np.pi, np.pi, 50)
    for theta in thetas:
        s_beta = f(theta)
        d_l_d_s = expected_gradient_loss_s(agent_dist, theta, s_beta, sigma, true_beta)
        emp_grad = empirical_gradient_loss_s(agent_dist, theta, s_beta, sigma, q, true_beta)
        emp_grad_s.append(emp_grad)
        grad_s.append(d_l_d_s)
    plt.plot(thetas, grad_s, label="expected")
    plt.plot(thetas, emp_grad_s, label="empirical")
    plt.xlabel("Theta (Corresponds to Beta)")
    plt.ylabel("dL/ds")
    plt.title("Beta vs. dL/ds")
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close()



