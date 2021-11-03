
def compute_score_bounds(beta):
    """Method that returns bounds on the highest and lowest possible scores that an agent can achieve.
    Assumes that agents take actions in [0, 1]^D

    Keyword arguments:
    beta -- modelparameters
    """
    min_score = 0.
    max_score = 0.
    for param in beta:
        if param >= 0:
            max_score += param
        else:
            min_score -= param
    return min_score, max_score
