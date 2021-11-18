from scipy.stats import bernoulli, norm, gaussian_kde
import numpy as np
from utils import convert_to_unit_vector, compute_score_bounds


class GradientEstimator:
    def __init__(
        self,
        agent_dist,
        theta,
        s,
        sigma,
        q,
        true_beta,
        perturbation_s_size,
        perturbation_theta_size,
    ):
        self.agent_dist = agent_dist
        self.sigma = sigma
        self.perturbations_s = (
            2 * bernoulli.rvs(p=0.5, size=agent_dist.n).reshape(agent_dist.n, 1) - 1
        )
        self.perturbations_theta = (
            2 * bernoulli.rvs(p=0.5, size=agent_dist.n).reshape(agent_dist.n, 1) - 1
        ) * perturbation_theta_size
        self.noise = norm.rvs(loc=0, scale=sigma, size=agent_dist.n).reshape(
            agent_dist.n, 1
        )
        self.theta = theta
        self.s = s
        self.q = q
        self.beta = convert_to_unit_vector(self.theta)
        self.true_beta = true_beta

        self.perturbation_s_size = perturbation_s_size
        self.perturbation_theta_size = perturbation_theta_size

    def set_perturbation_s_size(self):
        p_thetas = np.array([-1.0, 1.0]) * self.perturbation_theta_size
        p_ss = np.array([-1.0, 1.0]) * self.perturbation_s_size
        differences = []
        for p_theta in p_thetas:
            theta_perturbed = self.theta + p_theta
            beta_perturbed = convert_to_unit_vector(theta_perturbed)
            for p_s in p_ss:
                bounds = compute_score_bounds(beta_perturbed)
                s_perturbed = self.s + p_s
                if s_perturbed < bounds[0] or s_perturbed > bounds[1]:
                    difference = abs(
                        np.clip(s_perturbed, bounds[0], bounds[1]).item() - self.s
                    )
                    differences.append(difference)
        if len(differences) != 0:
            self.perturbation_s_size = min(differences)
        self.perturbations_s = self.perturbations_s * self.perturbation_s_size

    def get_scores(self):
        scores = []
        best_responses = {i: [] for i in range(self.agent_dist.n_types)}
        p_thetas = np.array([-1.0, 1.0]) * self.perturbation_theta_size
        p_ss = np.array([-1.0, 1.0]) * self.perturbation_s_size

        for agent_type in range(self.agent_dist.n_types):
            for p_theta in p_thetas:
                for p_s in p_ss:
                    theta_perturbed = self.theta + p_theta
                    beta_perturbed = convert_to_unit_vector(theta_perturbed)
                    s_perturbed = self.s + p_s
                    bounds = compute_score_bounds(beta_perturbed)
                    assert s_perturbed >= bounds[0] and s_perturbed <= bounds[1]
                    br = self.agent_dist.agents[agent_type].best_response(
                        beta_perturbed, s_perturbed, self.sigma
                    )
                    best_responses[agent_type].append(
                        {"p_theta": p_theta, "p_s": p_s, "br": br}
                    )

        for i in range(self.agent_dist.n):
            agent_type = self.agent_dist.n_agent_types[i]
            p_s = self.perturbations_s[i].item()
            p_theta = self.perturbations_theta[i].item()
            br_dics = best_responses[agent_type]
            for dic in br_dics:
                if dic["p_theta"] == p_theta and dic["p_s"] == p_s:
                    beta_perturbed = convert_to_unit_vector(self.theta + p_theta)
                    br = dic["br"]
                    scores.append(np.matmul(beta_perturbed.T, br).item() - p_s)
                    continue
        assert len(scores) == self.agent_dist.n
        scores = np.array(scores).reshape(self.agent_dist.n, 1)
        noisy_scores = scores + self.noise
        return noisy_scores

    def compute_gradients(self, noisy_scores, cutoff):
        Q_s = np.matmul(self.perturbations_s.T, self.perturbations_s)
        perturbations_theta = self.perturbations_theta.reshape(
            self.agent_dist.n, self.agent_dist.d - 1
        )
        Q_theta = np.matmul(perturbations_theta.T, perturbations_theta)

        # Compute loss
        treatments = noisy_scores >= cutoff
        etas = self.agent_dist.get_etas()
        loss_vector = treatments * np.array(
            [-np.matmul(self.true_beta.T, eta).item() for eta in etas]
        ).reshape(self.agent_dist.n, 1)

        gamma_loss_s = np.linalg.solve(
            Q_s, np.matmul(self.perturbations_s.T, loss_vector)
        ).item()
        gamma_loss_theta = np.linalg.solve(
            Q_theta, np.matmul(perturbations_theta.T, loss_vector)
        ).item()

        # Compute derivative of CDF
        indicators = noisy_scores < cutoff
        gamma_pi_s = np.linalg.solve(
            Q_s, np.matmul(self.perturbations_s.T, indicators)
        ).item()
        gamma_pi_theta = np.linalg.solve(
            Q_theta, np.matmul(perturbations_theta.T, indicators)
        ).item()

        return gamma_loss_s, gamma_loss_theta, gamma_pi_s, gamma_pi_theta, loss_vector

    def compute_density(self, scores, cutoff):
        kde = gaussian_kde(scores.flatten())
        return kde(cutoff).item()

    def compute_total_derivative(self):
        self.set_perturbation_s_size()
        # not sure if perturbations should be included for CDF scores may need to fix model params one
        scores = self.get_scores()
        cutoff = np.quantile(scores, self.q).item()

        (
            gamma_loss_s,
            gamma_loss_theta,
            gamma_pi_s,
            gamma_pi_theta,
            loss_vector,
        ) = self.compute_gradients(scores, cutoff)
        density_estimate = self.compute_density(scores, cutoff)

        gamma_s_theta = -(1 / (density_estimate + gamma_pi_s)) * gamma_pi_theta
        total_derivative = (gamma_loss_s * gamma_s_theta) + gamma_loss_theta

        dic = {
            "total_deriv": total_deriv,
            "partial_deriv_loss_s": gamma_loss_s,
            "partial_deriv_loss_theta": gamma_loss_theta,
            "partial_deriv_pi_s": gamma_pi_s,
            "partial_deriv_pi_theta": gamma_pi_theta,
            "partial_deriv_s_theta": gamma_s_theta,
            "density_estimate": density_estimate,
        }
        return dic, loss_vector.flatten()
