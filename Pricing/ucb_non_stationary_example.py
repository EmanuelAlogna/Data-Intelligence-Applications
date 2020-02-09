import matplotlib.pyplot as plt
from Non_Stationary_Environment import *
from SWUCB1_Learner import *

np.random.seed(10)
n_arms = 4


p = np.array([[0.17, 0.12, 0.27, 0.22],
              [0.62, 0.26, 0.52, 0.38],
              [0.40, 0.26, 0.12, 0.07],
              [0.73, 0.65, 0.54, 0.42]])

T = 368
n_experiments = 500
swucb_reward_per_experiment = []
ucb_reward_per_experiment = []
window_size = int(np.sqrt(T))


for e in range(0,n_experiments):
    print(e)
    sw_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p, horizon=T)
    ucb_learner = UCB1_Learner(n_arms=n_arms)

    swucb_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p, horizon=T)
    swucb_learner = SWUCB1_Learner(n_arms=n_arms, window_size=window_size)


    for t in range(0,T):
        pulled_arm = ucb_learner.pull_arm()
        reward = sw_env.round(pulled_arm)
        ucb_learner.update(pulled_arm, reward)

        pulled_arm = swucb_learner.pull_arm()
        reward = swucb_env.round(pulled_arm)
        swucb_learner.update(pulled_arm, reward)

    ucb_reward_per_experiment.append(ucb_learner.collected_rewards)
    swucb_reward_per_experiment.append(swucb_learner.collected_rewards)

ucb_instantaneus_regret = np.zeros(T)
swucb_instantaneus_regret = np.zeros(T)
n_phases = len(p)
phases_len = int(T/n_phases)
opt_per_phases = p.max(axis=1)
print(opt_per_phases)
opt_per_round = np.zeros(T)

for i in range(0, n_phases):
    opt_per_round[i*phases_len : (i+1)*phases_len] = opt_per_phases[i]
    ucb_instantaneus_regret[i * phases_len: (i + 1) * phases_len] = opt_per_phases[i] - np.mean(ucb_reward_per_experiment, axis=0)[i * phases_len: (i + 1) * phases_len]
    swucb_instantaneus_regret[i * phases_len: (i + 1) * phases_len] = opt_per_phases[i] - np.mean(swucb_reward_per_experiment, axis=0)[i * phases_len: (i + 1) * phases_len]




#In the first figure we show the reward
plt.figure(0)
plt.xlabel('t')
plt.ylabel('Reward')
plt.plot(np.mean(ucb_reward_per_experiment, axis=0), 'r')
plt.plot(np.mean(swucb_reward_per_experiment, axis=0), 'b')
plt.plot(opt_per_round, '--k')
plt.legend(['UCB1','SW-UCB1','Optimum'])
plt.show()


#In the second plot we show the regret
plt.figure(1)
plt.xlabel ('t')
plt.ylabel('Regret')
plt.plot(np.cumsum(ucb_instantaneus_regret), 'r')
plt.plot(np.cumsum(swucb_instantaneus_regret), 'b')
plt.legend(['UCB1','SW-UCB1'])
plt.show()
