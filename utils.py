import numpy as np
import tqdm
from scipy.interpolate import interp1d

def convert_to_polar_coordinates(beta):
    """Method that converts a D-dimensional unit vector to polar coordinates (D-1 - dimensional.)

    Keyword args:
    beta -- (D, 1) dimensional unit vector

    Returns:
    theta -- (D-1, 1) vector of angles
    """
    assert beta.shape[0] == 2, "Method only works for 2 dimensions now"

    theta = np.arctan2(beta[1], beta[0])
    return theta

def convert_to_unit_vector(theta):
    """Method that converts a polar coordinates to Euclidean unit vector.

    Keyword args:
    theta -- (D-1, 1) vector of angles

    Returns:
    beta -- (D, 1)  unit vector
    """
    assert isinstance(theta, float), "Method only works for float theta now"

    beta = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)
    return beta

def compute_continuity_noise(agent_dist):
    """Method that returns the standard deviation of the noise distribution for ensuring continuity.

    Keyword args:
    agent_dist -- AgentDistribution
    """
    gammas = agent_dist.get_gammas()

    min_eigenvalue = np.min(gammas)
    return np.sqrt(1 / (2 * min_eigenvalue * (np.sqrt(2 * np.pi * np.e)))) + 0.001


def compute_contraction_noise(agent_dist):
    """Method that returns the standard deviation of the noise distribution for ensuring contraction.

    Keyword args:
    agent_dist -- AgentDistribution
    """
    gammas = agent_dist.get_gammas()

    min_eigenvalue = np.min(gammas)
    return np.sqrt(1 / (min_eigenvalue * (np.sqrt(2 * np.pi * np.e)))) + 0.001


def compute_score_bounds(beta):
    """Method that returns bounds on the highest and lowest possible scores that an agent can achieve.
    Assumes that agents take actions in [0, 1]^2

    Keyword arguments:
    beta -- modelparameters
    """
    assert beta.shape[0] == 2, "Method does not work for beta with dim {}".format(
        beta.shape[0]
    )
    x_box = [
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.0, 0.0]),
    ]

    scores = [np.matmul(beta.T, x.reshape(2, 1)).item() for x in x_box]
    return min(scores), max(scores)

def spherical_coordinates(beta):
    assert beta.shape[0] == 2, "Method does not work for beta with dim {}".format(
        beta.shape[0]
    )

    return np.arctan2(beta[1], beta[0])

def fixed_point_interpolation_true_distribution(
    agent_dist, sigma, q, plot=False, savefig=None
):
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
    assert dim == 2, "Method does not work for dimension {}".format(dim)

    thetas = np.linspace(-np.pi, np.pi, 50)
    fixed_points = []

    # compute beta and fixed point for each theta
    print("Computing fixed points...")
    for theta in tqdm.tqdm(thetas):
        beta = np.array([np.cos(theta), np.sin(theta)]).reshape(dim, 1)
        fp = agent_dist.quantile_fixed_point_true_distribution(
            beta, sigma, q, plot=False
        )
        fixed_points.append(fp)

    f = interp1d(thetas, fixed_points, kind="linear")

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
