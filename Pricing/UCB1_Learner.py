from Learner import *
import math


class UCB1_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.numbers_of_selections = [0] * n_arms
        self.empirical_mean = [0] * n_arms

    def pull_arm(self):

        pulled_arm = 0
        max_upper_bound = 0

        for arm in range(0, self.n_arms):
            # if the arm arm has been already pulled once
            if self.numbers_of_selections[arm] > 0:
                total_counts = sum(self.numbers_of_selections)
                bound_length = math.sqrt(0.01*math.log(total_counts) / float(self.numbers_of_selections[arm]))
                upper_bound = self.empirical_mean[arm] + bound_length

            else:
                upper_bound = 1e400
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                pulled_arm = arm


        return pulled_arm

    def update(self, pulled_arm, reward):


        self.t += 1
        self.update_observations(pulled_arm, reward)

        self.numbers_of_selections[pulled_arm] = self.numbers_of_selections[pulled_arm] + 1
        n = self.numbers_of_selections[pulled_arm]

        old_empirical_mean = self.empirical_mean[pulled_arm]
        self.empirical_mean[pulled_arm] = ((n - 1) / float(n)) * old_empirical_mean + (1 / float(n)) * reward
