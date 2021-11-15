from scipy.stats import bernoulli, norm, gaussian_kde

class GradientEstimator:

    def __init__(self, agent_dist, theta, s, sigma, q, true_beta=None, perturbation_size=0.05):
        self.agent_dist = agent_dist
        self.sigma = sigma
        half = int(agent_dist.n/2)
        self.perturbations_s = (2 * bernoulli.rvs(p=0.5, size=half).reshape(half, 1) -1 ) * perturbation_size
        other_half = n - half
        self.perturbations_theta = (2 * bernoulli.rvs(p=0.5, size=other_half).reshape(other_half, 1) -1 ) * perturbation_size
        self.noise = norm.rvs(loc=0, scale=sigma, size=agent_dist.n)

        if true_beta is None:
            true_beta = np.zeros(beta.shape)
            true_beta[0] = 1.
        self.true_beta = true_beta
        self.theta = theta
        self.s = s
        self.q = q
        self.beta = convert_to_unit_vector(self.theta)
        self.bounds = compute_score_bounds(self.beta)

    def perturb_threshold_experiment(self):
        
        interpolators = []
        scores = []

        half = int(self.agent_dist.n/2)
        for agent in self.agent_dist.agents:
            interpolators.append(agent.br_score_function_s(self.beta, self.sigma))

        for i in range(half):
            s_perturbed = np.clip(self.s + self.perturbations_s[i], a_min=self.bounds[0], a_max=self.bounds[1])
            agent_type = self.agent_dist.n_agent_types[i]
            br_score = interpolators[agent_type](s_perturbed)
            scores.append(br_score.item())
            
        scores = np.array(scores).reshape(half, 1)
        noisy_scores = scores + self.noise[:half]
        perturbed_noisy_scores =  np.clip(noisy_scores - self.perturbations_s, a_min=self.bounds[0], a_max=self.bounds[1])
    
        #Compute loss
        treatments = perturbed_noisy_scores >= np.quantile(perturbed_noisy_scores, self.q)
        etas = self.agent_dist.get_etas()
        loss_vector = treatments * np.array([-np.matmul(self.true_beta.T, etas[i]).item() for i in range(half)]).reshape(half, 1)
        Q = np.matmul(self.perturbations_s.T, self.perturbations_s)
        gamma_loss_s = np.linalg.solve(Q, np.matmul(self.perturbations_s.T, loss_vector)).item()


        #Compute derivative of CDF
        indicators = noisy_scores <= self.s
        Q = np.matmul(self.perturbations_s.T, self.perturbations_s)
        gamma_pi_s = np.linalg.solve(Q, np.matmul(self.perturbations_s.T, indicators)).item()

        return gamma_loss_s, gamma_pi_s, perturbed_noisy_scores

    def perturb_model_params_experiment(self):
        
        beta = convert_to_unit_vector(self.theta)
        bounds = compute_score_bounds(self.beta)
        
        interpolators = []
        scores = []
        
        for agent in self.agent_dist.agents:
            f, _ = agent.br_score_function_beta(self.s, self.sigma)
            interpolators.append(f)

        half = int(self.agent_dist.n/2)
        other_half = self.agent_dist.n - half

        for i in range(other_half):
            theta_perturbed = self.theta + self.perturbations[i]
            if theta_perturbed < -np.pi:
                theta_perturbed += 2 * np.pi
            if theta_perturbed > np.pi:
                theta_perturbed -= 2 * np.pi
            agent_type = self.agent_dist.n_agent_types[half + i]
            br_score = interpolators[agent_type](theta_perturbed)
            scores.append(br_score.item())

        scores = np.array(scores).reshape(other_half, 1)
        noisy_scores = np.clip(scores + self.noise[half:], a_min=self.bounds[0], a_max=self.bounds[1])
        
        Q = np.matmul(perturbations_theta.T, perturbations_theta)
        perturbations_theta = self.perturbations_theta.reshape(other_half, self.agent_dist.d-1)

        #Compute loss
        treatments = noisy_scores >= np.quantile(noisy_scores, self.q)
        etas = self.agent_dist.get_etas()
        loss_vector = treatments * np.array([-np.matmul(self.true_beta.T, etas[i]).item() for i in range(half, self.agent_dist.n)]).reshape(other_half, 1)
        gamma_loss_theta = np.linalg.solve(Q, np.matmul(perturbations_theta.T, loss_vector))


        # Compute derivative of CDF
        indicators = noisy_scores <= self.s
        gamma_pi_theta = np.linalg.solve(Q, np.matmul(perturbations_theta.T, indicators)).item()

        return gamma_loss_theta, gamma_pi_theta, noisy_scores

    def estimate_density(self, noisy_scores_s, noisy_scores_theta):
        samples = np.concatenate((noisy_scores_s, noisy_scores_theta))
        kde = gaussian_kde(samples.flatten())
        return kde(self.s).item()

    def compute_total_derivative(self, theta, s):
        gamma_loss_s, gamma_pi_s, noisy_scores_s = self.perturb_threshold_experiment(theta, s)
        gamma_loss_theta, gamma_pi_theta, noisy_scores_theta = self.perturb_model_params_experiment(theta, s)
        density_estimate = self.compute_density(noisy_scores_s, noisy_scores_theta)

        gamma_s_theta = -(1/(density_estimate + gamma_pi_s)) * gamma_pi_theta
        total_derivative = gamma_loss_s * gamma_s_theta + gamma_loss_theta
        return total_derivative

