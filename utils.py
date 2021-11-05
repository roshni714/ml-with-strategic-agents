import numpy as np

def compute_continuity_noise(agent_dist):
    """Method that returns the standard deviation of the noise distribution for ensuring continuity.

    Keyword args:
    agent_dist -- AgentDistribution
    """
    gammas = agent_dist.get_gammas()

    min_eigenvalue = np.min(gammas)
    return np.sqrt(1/(2 * min_eigenvalue *(np.sqrt(2 * np.pi * np.e)))) + 0.001

def compute_contraction_noise(agent_dist):
    """Method that returns the standard deviation of the noise distribution for ensuring contraction.

    Keyword args:
    agent_dist -- AgentDistribution
    """
    gammas = agent_dist.get_gammas()

    min_eigenvalue = np.min(gammas)
    return np.sqrt(1/(min_eigenvalue *(np.sqrt(2 * np.pi * np.e)))) + 0.001


def compute_score_bounds(beta):
    """Method that returns bounds on the highest and lowest possible scores that an agent can achieve.
    Assumes that agents take actions in [0, 1]^2

    Keyword arguments:
    beta -- modelparameters
    """
    assert beta.shape[0] == 2, "Method does not work for beta with dim {}".format(beta.shape[0])
    x_box = [np.array([0., 1.]), np.array([1., 0.]), np.array([1., 1.]), np.array([0., 0.])]

    scores = [np.matmul(beta.T, x.reshape(2, 1)).item() for x in x_box]
    return min(scores), max(scores)
