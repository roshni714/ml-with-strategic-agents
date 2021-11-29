from scipy.stats import norm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from utils import compute_score_bounds
from agent import Agent


class AgentDistribution:
    """This is a class for representing a distribution over a finite number of agents.
    
    Keyword arguments:
    n -- number of agents in distribution
    d -- dimension of agent
    n_types -- number of agent types
    types -- optional argument: a dictionary of agent types of the form 
        {etas: (n_types, D, 1), gammas: (n_types, D, 1)}
    
    """

    def __init__(self, n=1000, d=2, n_types=50, types=None, prop=None):
        self.n = n
        self.d = d
        self.n_types = n_types
        self.types = types
        self.prop = prop
        if types is None:
            # Generate n_types agent types randomly
            etas = np.random.uniform(0.4, 0.6, size=n_types * d).reshape(n_types, d, 1)
            gammas = np.ones((n_types, d, 1)) * 8
        #            gammas = np.random.uniform(1.0, 2.0, size=n_types * d).reshape(
        else:
            etas = types["etas"]
            gammas = types["gammas"]
        if not prop:
            self.prop = np.ones(n_types) * (1 / n_types)
        else:
            self.prop = prop
        np.testing.assert_allclose(np.sum(self.prop), 1.0)
        self.n_agent_types = np.random.choice(
            list(range(self.n_types)), self.n, p=self.prop
        )

        # Create representative agents
        self.agents = []
        for i in range(n_types):
            self.agents.append(Agent(etas[i], gammas[i]))

    def get_etas(self):
        """Method that returns the etas for all agents in the distribution.

        Returns:
        etas -- (N, D, 1) array
        """
        etas = []
        for i in range(self.n):
            # get type of ith agent
            agent_type = self.n_agent_types[i]
            # get agent that has  type agent_type
            agent = self.agents[agent_type]
            # get eta
            etas.append(agent.eta)
        etas = np.array(etas).reshape(self.n, self.d, 1)
        return etas

    def get_gammas(self):
        """Method that returns the gammas for all agents in the distribution

        Returns:
        gammas -- (N, D, 1) array
        """
        gammas = []
        for i in range(self.n):
            # get type of ith agent
            agent_type = self.n_agent_types[i]
            # get agent that has  type agent_type
            agent = self.agents[agent_type]
            # get eta
            gammas.append(agent.gamma)
        gammas = np.array(gammas).reshape(self.n, self.d, 1)
        return gammas

    def best_response_distribution(self, beta, s, sigma):
        """This is a method that returns the best response of each agent type to a model and threshold.
        
        Keyword arguments:
        beta -- model parameters
        s -- threshold
        sigma -- standard deviation of noise distribution
        
        Returns:
        br -- a list of np.arrays
        """
        br = []
        for agent in self.agents:
            br.append(agent.best_response(beta, s, sigma))
        return br

    def br_gradient_beta_distribution(self, beta, s, sigma):
        """This is a method that returns the best response of each agent type to a model and threshold and the jacobian matrix.
        
        Keyword arguments:
        beta -- model parameters
        s -- threshold
        sigma -- standard deviation of noise distribution
        
        Returns:
        br -- a list of np.arrays of dimension (D, 1)
        jac -- a list of np.arrays of dimension (D, D)
        """
        br = []
        jac = []
        for agent in self.agents:
            b, j = agent.br_gradient_beta(beta, s, sigma)
            jac.append(j)
            br.append(b)
        return br, jac

    def br_gradient_theta_distribution(self, theta, s, sigma):
        """This is a method that returns the best response of each agent type to a model and threshold and the gradient wrt to theta.
        
        Keyword arguments:
        theta -- model parameters
        s -- threshold
        sigma -- standard deviation of noise distribution
        
        Returns
        br -- a list of np.arrays of dimension (D, 1)
        grad -- a list of np.arrays of dimension (D, 1)
        """
        br = []
        grad = []
        for agent in self.agents:
            b, j = agent.br_gradient_theta(theta, s, sigma)
            grad.append(j)
            br.append(b)
        return br, grad

    def br_gradient_s_distribution(self, beta, s, sigma):
        """This is a method that returns the best response of each agent type to a model and threshold and the derivative wrt to s.
        
        Keyword arguments:
        beta -- model parameters
        s -- threshold
        sigma -- standard deviation of noise distribution
        
        Returns:
        br -- a list of np.arrays of dimension (D, 1)
        deriv_s -- a list of np.arrays of dimension (D, 1)
        """
        br = []
        deriv_s = []
        for agent in self.agents:
            b, d = agent.br_gradient_s(beta, s, sigma)
            deriv_s.append(d)
            br.append(b)
        return br, deriv_s

    def best_response_score_distribution(self, beta, s, sigma):
        """This is a method that returns the score of the best response of each agent type to a model and threshold.
        
        Keyword arguments:
        beta -- model parameters (Nx1)
        s -- threshold (float)
        sigma -- standard deviation of noise distribution(float)
        
        Returns:
        br_dist -- a (n_types,) dimensional array
        """
        br_dist = [
            np.matmul(np.transpose(beta), x).item()
            for x in self.best_response_distribution(beta, s, sigma)
        ]
        return np.array(br_dist)

    def best_response_noisy_score_distribution(self, beta, s, sigma):
        """This is a method that returns the distribution over agent scores after noise has been added
        
        Keyword arguments:
        beta -- model parameters (Nx1)
        s -- threshold (float)
        sigma -- standard deviation of noise distribution(float)
        
        Returns:
        br_dist -- a (N, 1) dimensional array
        """
        bounds = compute_score_bounds(beta)
        noisy_scores = norm.rvs(loc=0.0, scale=sigma, size=self.n)
        br_dist = self.best_response_score_distribution(beta, s, sigma)

        n_br = br_dist[self.n_agent_types]
        noisy_scores += n_br
        #        noisy_scores = np.clip(noisy_scores, a_min=bounds[0], a_max=bounds[1])

        return noisy_scores.reshape(self.n, 1)

    def quantile_best_response(self, beta, s, sigma, q):
        """The method returns the qth quantile of the noisy score distribution.
        
        Keyword arguments:
        beta -- model parameters (Nx1)
        s -- threshold (float)
        sigma -- standard deviation of noise distribution(float)
        
        Returns:
        q_quantile -- qth quantile of the noisy score distribution (float)
        """
        noisy_scores = self.best_response_noisy_score_distribution(beta, s, sigma)
        q_quantile = np.quantile(noisy_scores, q)
        return q_quantile.item()

    def plot_quantile_best_response(self, beta, sigma, q):
        """This method plots the quantile of the noisy score distribution vs. thresholds.
        
        Keyword arguments:
        beta -- model parameters (Nx1)
        s -- threshold (float)
        sigma -- standard deviation of noise distribution(float)
        q -- quantile between 0 and 1 (float)
        """
        bounds = compute_score_bounds(beta)
        thresholds = np.linspace(bounds[0], bounds[1], 50)
        quantile_br = [
            self.quantile_best_response(beta, s, sigma, q) for s in thresholds
        ]

        plt.plot(thresholds, quantile_br)
        plt.xlabel("Thresholds")
        plt.ylabel("Quantile BR")
        plt.title("Quantile BR vs. Threshold")

    def quantile_mapping_vary_s(self, beta, sigma, q):
        """This method returns the quantile mapping function q(beta, s). 
        
        Keyword arguments:
        beta -- model parameters (Nx1)
        sigma -- standard deviation of noise distribution(float)
        q -- quantile between 0 and 1 (float)
        """
        bounds = compute_score_bounds(beta)
        thresholds = np.linspace(bounds[0], bounds[1], 50)
        quantile_map = []

        for s in tqdm.tqdm(thresholds):
            cdf_vals = []
            for r in thresholds:
                cdf_vals.append(self.best_response_cdf(beta, s, sigma, r))
            inverse_cdf_s = interp1d(cdf_vals, thresholds, kind="linear")
            quantile_map.append(inverse_cdf_s(q))
        q = interp1d(thresholds, quantile_map)
        return q

    def quantile_mapping_vary_beta(self, s, sigma, q):
        """This method returns the quantile mapping function q(beta, s). 
        
        Keyword arguments:
        s -- threshold (float)
        sigma -- standard deviation of noise distribution(float)
        q -- quantile between 0 and 1 (float)
        """
        thetas = np.linspace(-np.pi, np.pi, 50)
        quantile_map = []
        valid_theta = []
        for theta in tqdm.tqdm(thetas):
            cdf_vals = []
            beta = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)
            bounds = compute_score_bounds(beta)
            score_range = np.linspace(bounds[0], bounds[1], 50)
            if s >= bounds[0] and s <= bounds[1]:
                for r in score_range:
                    cdf_vals.append(self.best_response_cdf(beta, s, sigma, r))
                inverse_cdf_theta = interp1d(cdf_vals, score_range, kind="linear")
                plt.plot(cdf_vals, score_range)
                plt.xlabel("q")
                plt.ylabel("F^-1(q)")
                quantile_map.append(inverse_cdf_theta(q))
                valid_theta.append(theta)
        q = interp1d(valid_theta, quantile_map, kind="linear")
        return q, valid_theta

    def quantile_fixed_point_true_distribution(self, beta, sigma, q, plot=False):
        def compute_fs_s(s):
            cdf_val = 0.0
            for i, agent in enumerate(self.agents):
                cdf_val += (
                    norm.cdf(
                        s - np.matmul(beta.T, agent.best_response(beta, s, sigma)),
                        loc=0.0,
                        scale=sigma,
                    )
                    * self.prop[i]
                )
            return cdf_val.item()

        bounds = compute_score_bounds(beta)
        l = bounds[0]
        r = bounds[1]
        curr = (l + r) / 2
        val = compute_fs_s(curr)
        count = 0
        while abs(val - q) > 1e-10:
            if val > q:
                r = curr
            if val < q:
                l = curr

            curr = (l + r) / 2
            val = compute_fs_s(curr)
            count += 1
            if count > 20:
                break

        return curr

    def best_response_pdf(self, beta, s, sigma, r):
        bounds = compute_score_bounds(beta)
        if s < bounds[0]:
            return 0.0
        if s > bounds[1]:
            return 0.0

        pdf_val = 0.0

        for i, agent in enumerate(self.agents):
            pdf_val += (
                norm.pdf(
                    r - np.matmul(beta.T, agent.best_response(beta, s, sigma)),
                    loc=0.0,
                    scale=sigma,
                )
                * self.prop[i]
            )
        return pdf_val.item()

    def best_response_cdf(self, beta, s, sigma, r):
        bounds = compute_score_bounds(beta)
        if s < bounds[0]:
            return 0.0
        if s > bounds[1]:
            return 1.0

        cdf_val = 0.0
        for i, agent in enumerate(self.agents):
            cdf_val += (
                norm.cdf(
                    r - np.matmul(beta.T, agent.best_response(beta, s, sigma)),
                    loc=0.0,
                    scale=sigma,
                )
                * self.prop[i]
            )
        return cdf_val.item()

    def quantile_fixed_point_naive(self, beta, sigma, q, plot=False):
        bounds = compute_score_bounds(beta)
        thresholds = np.linspace(bounds[0], bounds[1], 500)
        quantile_br = [
            self.quantile_best_response(beta, s, sigma, q) for s in thresholds
        ]
        idx = np.argmin(np.abs(quantile_br - thresholds))
        fixed_point = thresholds[idx]

        if plot:
            plt.plot(thresholds, quantile_br)
            plt.xlabel("Thresholds")
            plt.ylabel("Quantile Best Response")
            plt.title("Quantile Best Response (from Empirical Distribution)")
            plt.show()
            plt.close()

        return fixed_point.item()

    def quantile_fixed_point_polyfit(self, beta, sigma, q, plot=False):
        """This method computes the fixed point of the quantile best response.
        
        This method computes the fixed point of the quantile best response mapping 
        by polynomial fitting of the quantile best response function.
        
        Keyword arguments:
        beta -- model parameters (Nx1)
        s -- threshold (float)
        sigma -- standard deviation of noise distribution(float)
        q -- quantile between 0 and 1 (float)
        plot -- optional plotting argument (False)
        
        Returns:
        fixed_point -- fixed point of the quantile best response (float)
        
        """
        bounds = compute_score_bounds(beta)
        thresholds = np.linspace(bounds[0], bounds[1], 50)
        quantile_br = [
            self.quantile_best_response(beta, s, sigma, q) for s in thresholds
        ]

        z = np.polyfit(thresholds.flatten(), quantile_br, 3)
        f = np.poly1d(z)
        granular_thresholds = np.linspace(bounds[0], bounds[1], 200)
        approx_quantile_best_response = f(granular_thresholds)
        idx = np.argmin(np.abs(approx_quantile_best_response - granular_thresholds))
        fixed_point = granular_thresholds[idx]

        if plot:
            plt.plot(granular_thresholds, approx_quantile_best_response)
            plt.plot(thresholds, quantile_br)
            plt.xlabel("Thresholds")
            plt.ylabel("Quantile Best Response")
            plt.title("Quantile Best Response Approximation")
            plt.show()
            plt.close()

        return fixed_point.item()

    def quantile_fixed_point_iteration(
        self, beta, sigma, q, maxiter=200, s0=0.5, plot=False
    ):
        """This method computes the fixed point of the quantile best response.
        
        This method computes the fixed point of the quantile best response mapping 
        by fixed point iteration. Not that this is not always guaranteed to converge.
        
        Keyword arguments:
        beta -- model parameters (Nx1)
        s -- threshold (float)
        sigma -- standard deviation of noise distribution(float)
        q -- quantile between 0 and 1 (float)
        
        Returns:
        fixed_point -- fixed point of the quantile best response (float)
        
        """
        bounds = compute_score_bounds(beta)
        thresholds = np.linspace(bounds[0], bounds[1], 50)

        all_s = [s0]
        s = s0
        for k in range(maxiter):
            new_s = self.quantile_best_response(beta, s, sigma, q)
            all_s.append(new_s)
            s = new_s
            if plot and k % 50 == 0 and k > 0:
                plt.plot(list(range(len(all_s))), all_s)
                plt.show()
                plt.close()
        return s

    def resample(self):
        self.n_agent_types = np.random.choice(
            list(range(self.n_types)), self.n, p=self.prop
        )


if __name__ == "__main__":

    agent_dist = AgentDistribution()
    etas = agent_dist.get_etas()
    gammas = agent_dist.get_gammas()
    etas2 = agent_dist.get_etas()
    print(etas.shape)
    print(gammas.shape)
