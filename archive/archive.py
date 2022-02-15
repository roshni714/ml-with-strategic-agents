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
    assert dim == 2, "Method does not work for dimension {}".format(dim)

    beta = convert_to_unit_vector(theta)
    if true_beta is None:
        true_beta = np.zeros((agent_dist.d, 1))
        true_beta[0] = 1.0
    bounds = compute_score_bounds(beta)

    true_scores = np.array(
        [np.matmul(true_beta.T, agent.eta).item() for agent in agent_dist.agents]
    ).reshape(agent_dist.n_types, 1)

    br_dist, grad_theta_dist = agent_dist.br_gradient_theta_distribution(
        theta, s, sigma
    )
    z = s - np.array([np.matmul(beta.T, x) for x in br_dist]).reshape(len(br_dist), 1)
    prob = norm.pdf(z, loc=0.0, scale=sigma)

    dbeta_dtheta = np.array([-np.sin(theta), np.cos(theta)]).reshape(2, 1)
    first_term = np.array(
        [
            np.matmul(grad_theta_dist[i].T, beta).item()
            for i in range(len(grad_theta_dist))
        ]
    ).reshape(agent_dist.n_types, 1)
    second_term = np.array(
        [
            np.matmul(br_dist[i].T, dbeta_dtheta).item()
            for i in range(len(grad_theta_dist))
        ]
    ).reshape(agent_dist.n_types, 1)
    total = first_term + second_term

    res = -prob * total * true_scores * agent_dist.prop.reshape(agent_dist.n_types, 1)
    dl_dtheta = np.sum(res).item()

    return dl_dtheta


def empirical_gradient_loss_theta(
    agent_dist, theta, s, sigma, q, true_beta=None, perturbation_size=0.1
):
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
        true_beta[0] = 1.0

    perturbations = (
        2
        * bernoulli.rvs(p=0.5, size=agent_dist.n * (agent_dist.d - 1)).reshape(
            agent_dist.n, agent_dist.d - 1, 1
        )
        - 1
    ) * perturbation_size
    scores = []

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
    noise = norm.rvs(loc=0.0, scale=sigma, size=agent_dist.n).reshape(agent_dist.n, 1)
    noisy_scores = np.clip(scores + noise, a_min=bounds[0], a_max=bounds[1])

    # Compute loss
    treatments = noisy_scores >= np.quantile(noisy_scores, q)
    loss_vector = treatments * np.array(
        [-np.matmul(true_beta.T, eta).item() for eta in agent_dist.get_etas()]
    ).reshape(agent_dist.n, 1)

    perturbations = perturbations.reshape(agent_dist.n, agent_dist.d - 1)
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
        grad_theta.append(grad)
        emp_grad = empirical_gradient_loss_theta(
            agent_dist, theta, s_beta, sigma, q, true_beta
        )
        emp_grad_theta.append(emp_grad)

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
