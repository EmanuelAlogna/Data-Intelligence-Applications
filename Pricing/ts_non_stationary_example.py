import matplotlib.pyplot as plt
from Non_Stationary_Environment import *
from SWTS_Learner import *
np.random.seed(10)
n_arms = 4


p = np.array([[0.17, 0.12, 0.27, 0.22],
              [0.62, 0.26, 0.52, 0.38],
              [0.40, 0.26, 0.12, 0.07],
              [0.73, 0.65, 0.54, 0.42]])

p = p*0.2
T = 10000
n_experiments = 50
swts_reward_per_experiment = []
ts_reward_per_experiment = []
window_size = int(np.sqrt(T))


for e in range(0,n_experiments):
    print(e)
    ts_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p, horizon=T)
    ts_learner = TS_Learner(n_arms=n_arms)

    swts_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p, horizon=T)
    swts_learner = SWTS_Learner(n_arms=n_arms, window_size=window_size)


    for t in range(0,T):
        pulled_arm = ts_learner.pull_arm()
        reward = ts_env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        pulled_arm = swts_learner.pull_arm()
        reward = swts_env.round(pulled_arm)
        swts_learner.update(pulled_arm, reward)

    ts_reward_per_experiment.append(ts_learner.collected_rewards)
    swts_reward_per_experiment.append(swts_learner.collected_rewards)

ts_instantaneus_regret = np.zeros(T)
swts_instantaneus_regret = np.zeros(T)
n_phases = len(p)
phases_len = int(T/n_phases)
opt_per_phases = p.max(axis=1)
opt_per_round = np.zeros(T)

for i in range(0, n_phases):
    opt_per_round[i*phases_len : (i+1)*phases_len] = opt_per_phases[i]
    ts_instantaneus_regret[i*phases_len : (i+1)*phases_len] = opt_per_phases[i] - np.mean(ts_reward_per_experiment,axis=0)[i*phases_len : (i+1)*phases_len]
    swts_instantaneus_regret[i*phases_len : (i+1)*phases_len] = opt_per_phases[i] - np.mean(swts_reward_per_experiment,axis=0)[i*phases_len : (i+1)*phases_len]




#In the first figure we show the reward
plt.figure(0)
plt.xlabel('t')
plt.ylabel('Reward')
plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'r')
plt.plot(np.mean(swts_reward_per_experiment, axis=0), 'b')
plt.plot(opt_per_round, '--k')
plt.legend(['TS','SW-TS','Optimum'])
plt.show()


#In the second plot we show the regret
plt.figure(1)
plt.xlabel ('t')
plt.ylabel('Regret')
plt.plot(np.cumsum(ts_instantaneus_regret), 'r')
plt.plot(np.cumsum(swts_instantaneus_regret), 'b')
plt.legend(['TS','SW-TS'])
plt.show()
