
def expected_partial_loss_partial_beta(agent_dist, theta, sigma, f, true_beta=None):
    """Method computes partial L(beta)/partial beta.

    Keyword args:
    agent_dist -- AgentDistribution
    beta -- model parameters
    sigma -- standard deviation of noise distribution
    f -- function that maps arctan(beta[1]/beta[0]) -> s_beta (fixed point)
    true_beta -- optional ideal model
    
    
    Returns:
    d_l_d_beta -- expected gradient of policy loss at beta

    """
    dim = agent_dist.d
    assert dim==2, "Method does not work for dimension {}".format(dim)
    
    beta = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)
    if true_beta is None:
        true_beta = np.zeros(beta.shape)
        true_beta[0] = 1.
    bounds = compute_score_bounds(beta)
    s = np.clip(f(theta), a_min=bounds[0], a_max = bounds[1])
    true_scores = np.array([-np.matmul(true_beta.T, agent.eta).item() for agent in agent_dist.agents])
    br_dist, jacobian_dist = agent_dist.jacobian_beta_distribution(beta, s, sigma)
    z = s - br_dist
    prob = norm.pdf(z, loc=0., scale=sigma)
    vec = [np.matmul(beta.T, jacobian_dist[i]) - br_dist[i].T for i in range(len(br_dist))]

    res = vec * true_scores * agent_dist.prop
    return np.mean(res, axis=1)

 
