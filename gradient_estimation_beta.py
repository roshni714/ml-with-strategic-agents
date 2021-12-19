from scipy.stats import bernoulli, norm, gaussian_kde
import numpy as np
from utils import convert_to_unit_vector, compute_score_bounds, keep_theta_in_bounds
import itertools


class GradientEstimator:
    def __init__(
        self,
        agent_dist,
        beta,
        s,
        sigma,
        q,
        true_beta,
        perturbation_s_size,
        perturbation_beta_size,
    ):
        self.agent_dist = agent_dist
        self.sigma = sigma

        self.perturbations_s = (
            2 * bernoulli.rvs(p=0.5, size=agent_dist.n).reshape(agent_dist.n, 1) - 1
        )
        self.perturbations_beta = (
            2
            * bernoulli.rvs(p=0.5, size=agent_dist.n * beta.shape[0]).reshape(
                agent_dist.n, beta.shape[0]
            )
            - 1
        )
        self.noise = norm.rvs(loc=0.0, scale=sigma, size=agent_dist.n).reshape(
            agent_dist.n, 1
        )
        self.beta = beta
        self.s = s
        self.q = q
        self.true_beta = true_beta
        etas = agent_dist.get_etas()
        self.true_scores = np.array(
            [-np.matmul(self.true_beta.T, eta).item() for eta in etas]
        ).reshape(agent_dist.n, 1)
        self.perturbation_s_size = perturbation_s_size
        self.perturbation_beta_size = perturbation_beta_size

    def get_best_responses(self):
        best_responses = {i: [] for i in range(self.agent_dist.n_types)}
        p_betas = np.array(
            list(itertools.product([-1.0, 1.0], repeat=self.beta.shape[0]))
        )
        p_ss = np.array(list(itertools.product([-1.0, 1.0], repeat=1)))

        for agent_type in range(self.agent_dist.n_types):
            for p_beta in p_betas:
                for p_s in p_ss:
                    beta_perturbed = self.beta + (
                        p_beta.reshape(self.beta.shape) * self.perturbation_beta_size
                    )
                    s_perturbed = self.s + (p_s * self.perturbation_s_size).item()
                    br = self.agent_dist.agents[agent_type].best_response(
                        beta_perturbed, s_perturbed, self.sigma
                    )
                    best_responses[agent_type].append(
                        {"p_s": p_s, "p_beta": p_beta, "br": br}
                    )

        return best_responses

    def get_scores(self, best_responses):
        unperturbed_scores = []
        beta_perturbed_scores = []
        scores = []
        for i in range(self.agent_dist.n):
            agent_type = self.agent_dist.n_agent_types[i]
            p_beta = self.perturbations_beta[i]
            p_s = self.perturbations_s[i]
            br_dics = best_responses[agent_type]
            for dic in br_dics:
                if np.all(dic["p_beta"] == p_beta) and dic["p_s"] == p_s:
                    beta_perturbed = self.beta + (
                        np.array(p_beta).reshape(self.beta.shape)
                        * self.perturbation_beta_size
                    )
                    br = dic["br"]
                    scores.append(
                        np.matmul(beta_perturbed.T, br).item()
                        - (p_s.item() * self.perturbation_s_size)
                    )
                    beta_perturbed_scores.append(np.matmul(beta_perturbed.T, br).item())
                    unperturbed_scores.append(np.matmul(self.beta.T, br).item())
                    continue
        scores = np.array(scores).reshape(self.agent_dist.n, 1)
        #        unperturbed_scores = np.array(unperturbed_scores).reshape(self.agent_dist.n, 1)
        scores = self.noise + scores
        unperturbed_scores = self.noise + np.array(unperturbed_scores).reshape(
            self.agent_dist.n, 1
        )
        beta_perturbed_scores = self.noise + np.array(beta_perturbed_scores).reshape(
            self.agent_dist.n, 1
        )
        return scores, unperturbed_scores, beta_perturbed_scores

    def compute_gradients(
        self, scores, unperturbed_scores, beta_perturbed_scores, cutoff
    ):
        p_s = self.perturbations_s * self.perturbation_s_size
        p_beta = self.perturbations_beta * self.perturbation_beta_size
        Q_s = np.matmul(p_s.T, p_s)
        p_beta = p_beta.reshape(self.agent_dist.n, self.agent_dist.d)
        Q_beta = np.matmul(p_beta.T, p_beta)

        # Compute loss
        treatments = scores > cutoff
        loss_vector = treatments * self.true_scores

        gamma_loss_s = np.linalg.solve(Q_s, np.matmul(p_s.T, loss_vector)).reshape(1, 1)
        gamma_loss_beta = np.linalg.solve(
            Q_beta, np.matmul(p_beta.T, loss_vector)
        ).reshape(self.agent_dist.d, 1)

        indicators_beta = beta_perturbed_scores > cutoff
        indicators_s = unperturbed_scores > cutoff
        gamma_pi_s = -np.linalg.solve(Q_s, np.matmul(p_s.T, indicators_s)).reshape(1, 1)
        gamma_pi_beta = -np.linalg.solve(
            Q_beta, np.matmul(p_beta.T, indicators_beta)
        ).reshape(self.agent_dist.d, 1)

        return gamma_loss_s, gamma_loss_beta, gamma_pi_s, gamma_pi_beta, loss_vector

    def compute_density(self, scores, cutoff):
        kde = gaussian_kde(scores.flatten())
        return kde(cutoff).reshape(1, 1)

    def compute_total_derivative(self):
        br = self.get_best_responses()
        scores, unperturbed_scores, beta_perturbed_scores = self.get_scores(br)
        cutoff = np.quantile(scores, self.q).item()

        (
            gamma_loss_s,
            gamma_loss_beta,
            gamma_pi_s,
            gamma_pi_beta,
            loss_vector,
        ) = self.compute_gradients(
            scores, unperturbed_scores, beta_perturbed_scores, cutoff
        )
        density_estimate = self.compute_density(unperturbed_scores, cutoff)

        gamma_s_beta = -(1 / (density_estimate + gamma_pi_s)) * gamma_pi_beta
        total_deriv = (gamma_loss_s * gamma_s_beta) + gamma_loss_beta

        assert total_deriv.shape == (self.agent_dist.d, 1)

        dic = {
            "total_deriv": total_deriv,
            "partial_deriv_loss_s": gamma_loss_s,
            "partial_deriv_loss_beta": gamma_loss_beta,
            "partial_deriv_pi_s": gamma_pi_s,
            "partial_deriv_pi_beta": gamma_pi_beta,
            "partial_deriv_s_beta": gamma_s_beta,
            "density_estimate": density_estimate,
            "loss": loss_vector.mean().item(),
        }
        return dic
