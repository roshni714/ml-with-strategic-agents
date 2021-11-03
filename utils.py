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
    Assumes that agents take actions in [0, 1]^D

    Keyword arguments:
    beta -- modelparameters
    """
    min_score = 0.0
    max_score = 0.0
    for param in beta:
        if param >= 0:
            max_score += param
        else:
            min_score -= param
    return min_score, max_score
