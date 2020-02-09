from UCB1_Learner import *

class SWUCB1_Learner(UCB1_Learner):
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size
        self.selections_windowed = [0.0] * n_arms

    def pull_arm(self):
        pulled_arm = 0
        max_upper_bound = 0
        bound_length = 0

        for arm in range(0, self.n_arms):
            if self.selections_windowed[arm] > 0:
                bound_length = math.sqrt(0.01*math.log(self.t) / float(self.selections_windowed[arm]))
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
        self.pulled_arms = self.pulled_arms.astype(int)
        temp = np.bincount(self.pulled_arms[-self.window_size:], minlength=self.n_arms)
        self.selections_windowed = temp
        num_selections_pulled_arm = self.selections_windowed[pulled_arm]
        overall = np.sum(self.rewards_per_arm[pulled_arm][-num_selections_pulled_arm:])
        size = len(self.rewards_per_arm[pulled_arm][-num_selections_pulled_arm:])
        self.empirical_mean[pulled_arm] = overall/size
