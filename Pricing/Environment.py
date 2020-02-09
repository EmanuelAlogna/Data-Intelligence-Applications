import numpy as np

class Environment():
    def __init__(self, n_arms, probabilities):
        # n_arms: number of arms
        # prbabilities: probability distribution of the arm reward
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm):
        # returns a random binomial reward associated to the pulled arm
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward



