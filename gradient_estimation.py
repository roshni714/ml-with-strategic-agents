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
            2 * bernoulli.rvs(p=0.5, size=int(agent_dist.n/2)).reshape(int(agent_dist.n/2), 1) - 1
        ) * perturbation_s_size
        self.perturbations_theta = (
            2 * bernoulli.rvs(p=0.5, size=int(agent_dist.n/2)).reshape(int(agent_dist.n/2), 1) - 1
        ) * perturbation_theta_size
        self.noise = norm.rvs(loc=0, scale=sigma, size=agent_dist.n).reshape(
            agent_dist.n, 1
        )
 #       print("perturbation theta size", perturbation_theta_size)
        self.theta = theta
        self.s = s
        self.q = q
        self.beta = convert_to_unit_vector(self.theta)
        self.true_beta = true_beta

        self.perturbation_s_size = perturbation_s_size
        self.perturbation_theta_size = perturbation_theta_size

 #        print("perturbation s size", self.perturbation_s_size)
    def get_theta_perturbed_scores(self):
        scores = []
        best_responses = {i: [] for i in range(self.agent_dist.n_types)}
        p_thetas = np.array([-1.0, 1.0]) * self.perturbation_theta_size

        for agent_type in range(self.agent_dist.n_types):
            for p_theta in p_thetas:
                theta_perturbed = self.theta + p_theta
                beta_perturbed = convert_to_unit_vector(theta_perturbed)
                br = self.agent_dist.agents[agent_type].best_response(
                        beta_perturbed, self.s, self.sigma
                    )
                best_responses[agent_type].append(
                        {"p_theta": p_theta, "br": br}
                    )

        unperturbed_scores = []
        for i in range(int(self.agent_dist.n/2)):
            agent_type = self.agent_dist.n_agent_types[i]
            p_theta = self.perturbations_theta[i].item()
            br_dics = best_responses[agent_type]
            for dic in br_dics:
                if dic["p_theta"] == p_theta:
                    beta_perturbed = convert_to_unit_vector(self.theta + p_theta)
                    br = dic["br"]
                    scores.append(np.matmul(beta_perturbed.T, br).item())
                    unperturbed_scores.append(np.matmul(self.beta.T, br).item())
                    continue
        scores = np.array(scores).reshape(int(self.agent_dist.n/2), 1)
        noisy_scores = scores + self.noise[:int(self.agent_dist.n/2)]
        noisy_unperturbed_scores = np.array(unperturbed_scores).reshape(int(self.agent_dist.n/2), 1) + self.noise[:int(self.agent_dist.n/2)]
        return noisy_scores, noisy_unperturbed_scores

    def get_s_perturbed_scores(self):
        scores = []
        best_responses = {i: [] for i in range(self.agent_dist.n_types)}
        p_ss = np.array([-1.0, 1.0]) * self.perturbation_s_size

        for agent_type in range(self.agent_dist.n_types):
            for p_s in p_ss:
                s_perturbed = self.s + p_s
                br = self.agent_dist.agents[agent_type].best_response(
                        self.beta, s_perturbed, self.sigma
                    )
                best_responses[agent_type].append(
                        {"p_s": p_s, "br": br}
                    )

        unperturbed_scores = [] 
        for i in range(int(self.agent_dist.n/2), self.agent_dist.n):
            agent_type = self.agent_dist.n_agent_types[i]
            p_s = self.perturbations_s[i - int(self.agent_dist.n/2)].item()
            br_dics = best_responses[agent_type]
            for dic in br_dics:
                if dic["p_s"] == p_s:
                    br = dic["br"]
                    scores.append(np.matmul(self.beta.T, br).item() - p_s)
                    unperturbed_scores.append(np.matmul(self.beta.T, br).item())
                    continue
        scores = np.array(scores).reshape(int(self.agent_dist.n/2), 1)
        noisy_scores = scores + self.noise[int(self.agent_dist.n/2):]
        noisy_unperturbed_scores = np.array(unperturbed_scores).reshape(int(self.agent_dist.n/2), 1) + self.noise[int(self.agent_dist.n/2):] 
        return noisy_scores, noisy_unperturbed_scores

    def compute_gradients(self, scores_s_perturbed, scores_theta_perturbed, cutoff):
        Q_s = np.matmul(self.perturbations_s.T, self.perturbations_s)
        perturbations_theta = self.perturbations_theta.reshape(
            int(self.agent_dist.n/2), self.agent_dist.d - 1
        )
        Q_theta = np.matmul(perturbations_theta.T, perturbations_theta)

        # Compute loss
        treatments_s = scores_s_perturbed >= cutoff 
        treatments_theta = scores_theta_perturbed >= cutoff
        etas = self.agent_dist.get_etas()
        loss_vector_s = treatments_s * np.array(
            [-np.matmul(self.true_beta.T, etas[i]).item() for i in range(int(self.agent_dist.n/2))]
        ).reshape(int(self.agent_dist.n/2), 1)

        loss_vector_theta = treatments_theta * np.array(
            [-np.matmul(self.true_beta.T, etas[i]).item() for i in range(int(self.agent_dist.n/2), self.agent_dist.n)]
        ).reshape(int(self.agent_dist.n/2), 1)

        gamma_loss_s = np.linalg.solve(
            Q_s, np.matmul(self.perturbations_s.T, loss_vector_s)
        ).item()
        gamma_loss_theta = np.linalg.solve(
            Q_theta, np.matmul(perturbations_theta.T, loss_vector_theta)
        ).item()

        # Compute derivative of CDF
        indicators_s =   scores_s_perturbed + self.perturbations_s < cutoff #remove perturbations for estimating CDF
        indicators_theta = scores_theta_perturbed < cutoff
        gamma_pi_s = np.linalg.solve(
            Q_s, np.matmul(self.perturbations_s.T, indicators_s)
        ).item()
        gamma_pi_theta = np.linalg.solve(
            Q_theta, np.matmul(perturbations_theta.T, indicators_theta)
        ).item()

        loss_vector = np.concatenate((loss_vector_s, loss_vector_theta)).reshape(self.agent_dist.n, 1)

        return gamma_loss_s, gamma_loss_theta, gamma_pi_s, gamma_pi_theta, loss_vector

    def compute_density(self, scores, cutoff):
        kde = gaussian_kde(scores.flatten())
        return kde(cutoff).item()

    def compute_total_derivative(self):
 #       self.set_perturbation_s_size()
        # not sure if perturbations should be included for CDF scores may need to fix model params one
        scores_theta_perturbed, scores_theta_unperturbed = self.get_theta_perturbed_scores()
        scores_s_perturbed, scores_s_unperturbed = self.get_s_perturbed_scores()
        scores = np.concatenate((scores_s_unperturbed, scores_theta_unperturbed)).reshape(self.agent_dist.n, 1)

        cutoff = np.quantile(scores, self.q).item()

        (
            gamma_loss_s,
            gamma_loss_theta,
            gamma_pi_s,
            gamma_pi_theta,
            loss_vector,
        ) = self.compute_gradients(scores_s_perturbed, scores_theta_perturbed, cutoff)
        density_scores = np.concatenate((scores_s_unperturbed, scores_theta_unperturbed)).reshape(self.agent_dist.n, 1)
        density_estimate = self.compute_density(density_scores, cutoff)

        gamma_s_theta = -(1 / (density_estimate + gamma_pi_s)) * gamma_pi_theta
        total_deriv = (gamma_loss_s * gamma_s_theta) + gamma_loss_theta

        dic = {
            "total_deriv": total_deriv,
            "partial_deriv_loss_s": gamma_loss_s,
            "partial_deriv_loss_theta": gamma_loss_theta,
            "partial_deriv_pi_s": gamma_pi_s,
            "partial_deriv_pi_theta": gamma_pi_theta,
            "partial_deriv_s_theta": gamma_s_theta,
            "density_estimate": density_estimate,
            "losses": loss_vector.flatten()
        }
        return dic

"""
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

"""
