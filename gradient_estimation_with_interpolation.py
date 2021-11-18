from scipy.stats import bernoulli, norm, gaussian_kde
import numpy as np
from utils import convert_to_unit_vector, compute_score_bounds


class GradientEstimatorWithInterpolation:
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
        half = int(agent_dist.n / 2)
        self.perturbations_s = (
            2 * bernoulli.rvs(p=0.5, size=half).reshape(half, 1) - 1
        ) * perturbation_s_size
        other_half = agent_dist.n - half
        self.perturbations_theta = (
            2 * bernoulli.rvs(p=0.5, size=other_half).reshape(other_half, 1) - 1
        ) * perturbation_theta_size
        self.noise = norm.rvs(loc=0, scale=sigma, size=agent_dist.n).reshape(
            agent_dist.n, 1
        )
        self.theta = theta
        self.s = s
        self.q = q
        self.beta = convert_to_unit_vector(self.theta)
        self.bounds = compute_score_bounds(self.beta)
        self.true_beta = true_beta

    def get_br_threshold_experiment(self):

        interpolators = []
        scores = []

        half = int(self.agent_dist.n / 2)
        for agent in self.agent_dist.agents:
            interpolators.append(agent.br_score_function_s(self.beta, self.sigma))

        for i in range(half):
            s_perturbed = np.clip(
                self.s + self.perturbations_s[i],
                a_min=self.bounds[0],
                a_max=self.bounds[1],
            )
            agent_type = self.agent_dist.n_agent_types[i]
            br_score = interpolators[agent_type](s_perturbed)
            scores.append(br_score.item())

        scores = np.array(scores).reshape(half, 1)
        noisy_scores = scores + self.noise[:half]
        perturbed_noisy_scores = noisy_scores - self.perturbations_s

        return perturbed_noisy_scores

    def compute_gradients_threshold(self, noisy_scores, cutoff):
        half = int(self.agent_dist.n / 2)
        Q = np.matmul(self.perturbations_s.T, self.perturbations_s)

        # Compute loss
        treatments = noisy_scores >= cutoff
        etas = self.agent_dist.get_etas()
        loss_vector = treatments * np.array(
            [-np.matmul(self.true_beta.T, etas[i]).item() for i in range(half)]
        ).reshape(half, 1)

        gamma_loss_s = np.linalg.solve(
            Q, np.matmul(self.perturbations_s.T, loss_vector)
        ).item()

        # Compute derivative of CDF
        indicators = noisy_scores < cutoff
        gamma_pi_s = np.linalg.solve(
            Q, np.matmul(self.perturbations_s.T, indicators)
        ).item()

        return gamma_loss_s, gamma_pi_s, loss_vector

    def get_br_model_params_experiment(self):

        interpolators = []
        scores = []

        for agent in self.agent_dist.agents:
            f, _ = agent.br_score_function_beta(self.s, self.sigma)
            interpolators.append(f)

        half = int(self.agent_dist.n / 2)
        other_half = self.agent_dist.n - half

        for i in range(other_half):
            theta_perturbed = self.theta + self.perturbations_theta[i]
            if theta_perturbed < -np.pi:
                theta_perturbed += 2 * np.pi
            if theta_perturbed > np.pi:
                theta_perturbed -= 2 * np.pi
            agent_type = self.agent_dist.n_agent_types[half + i]
            br_score = interpolators[agent_type](theta_perturbed)
            scores.append(br_score.item())

        scores = np.array(scores).reshape(other_half, 1)
        noisy_scores = scores + self.noise[half:]
        return noisy_scores

    def compute_gradients_model_params(self, noisy_scores, cutoff):
        half = int(self.agent_dist.n / 2)
        other_half = self.agent_dist.n - half

        perturbations_theta = self.perturbations_theta.reshape(
            other_half, self.agent_dist.d - 1
        )
        Q = np.matmul(perturbations_theta.T, perturbations_theta)

        # Compute loss
        treatments = noisy_scores >= cutoff
        etas = self.agent_dist.get_etas()
        loss_vector = treatments * np.array(
            [
                -np.matmul(self.true_beta.T, etas[i]).item()
                for i in range(half, self.agent_dist.n)
            ]
        ).reshape(other_half, 1)
        gamma_loss_theta = np.linalg.solve(
            Q, np.matmul(perturbations_theta.T, loss_vector)
        ).item()

        # Compute derivative of CDF
        indicators = noisy_scores < cutoff
        gamma_pi_theta = np.linalg.solve(
            Q, np.matmul(perturbations_theta.T, indicators)
        ).item()

        return gamma_loss_theta, gamma_pi_theta, loss_vector

    def compute_density(self, scores, cutoff):
        kde = gaussian_kde(scores)
        return kde(cutoff).item()

    def compute_total_derivative(self):
        # not sure if perturbations should be included for CDF scores may need to fix model params one
        final_noisy_scores_s = self.get_br_threshold_experiment()
        final_noisy_scores_theta = self.get_br_model_params_experiment()

        scores = np.concatenate(
            (final_noisy_scores_s, final_noisy_scores_theta)
        ).flatten()
        cutoff = np.quantile(scores, self.q).item()

        gamma_loss_s, gamma_pi_s, loss_vector_s = self.compute_gradients_threshold(
            final_noisy_scores_s, cutoff
        )
        (
            gamma_loss_theta,
            gamma_pi_theta,
            loss_vector_theta,
        ) = self.compute_gradients_model_params(final_noisy_scores_theta, cutoff)
        density_estimate = self.compute_density(scores, cutoff)

        gamma_s_theta = -(1 / (density_estimate + gamma_pi_s)) * gamma_pi_theta
        total_derivative = (gamma_loss_s * gamma_s_theta) + gamma_loss_theta

        losses = np.concatenate((loss_vector_s, loss_vector_theta)).flatten()
        return total_derivative, losses
