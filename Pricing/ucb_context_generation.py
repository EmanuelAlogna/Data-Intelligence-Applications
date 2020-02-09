import matplotlib.pyplot as plt
from Environment import *
from UCB1_Learner import *
import pandas as pd
import math
np.random.seed(10)
n_arms = 4

p = np.array([[0.77, 0.64, 0.54, 0.46],[0.68, 0.57, 0.51, 0.38],[0.74, 0.62, 0.45, 0.36]])
opt = np.max(np.mean(p,axis=0))

print(opt)
T = 365

n_experiments = 250
ts_rewards_per_experiment = []
gr_rewards_per_experiment = []
z = pd.DataFrame(columns=['Gender','Age','Arm','Reward'])

arms = np.array([0,1,2,3])

def split_on_age(z,delta=0.1):

    G = compute_profit(z,delta)

    #under has age == 0
    z_l = z.loc[z['Age'] == 0]
    z_r = z.loc[z['Age'] == 1]
    G_l = compute_profit(z_l,delta= delta/4)
    G_r = compute_profit(z_r,delta= delta/4)

    return (G_r + G_l - G > 0)

def split_on_gender(z,delta=0.05):
    z_under = z.loc[z['Age'] == 0]

    G = compute_profit(z_under,delta)

    z_l = z_under.loc[z['Gender'] == 0]
    z_r = z_under.loc[z['Gender'] == 1]
    G_l = compute_profit(z_l, delta=delta / 4)
    G_r = compute_profit(z_r, delta=delta / 4)
    return (G_r + G_l - G > 0)

def compute_profit(z,delta):
    profit_per_arm = []
    for arm in arms:
        n = z.shape[0]
        z1 = z.loc[z['Arm'] == arm]
        if z1.shape[0] == 0:
            profit_per_arm.append(-1e400)
            continue
        p = z.shape[0] / n
        p_confidence_bound = math.sqrt(-(math.log(delta/2)/(2*n)))
        p_lower_bound = p - p_confidence_bound

        arm_reward = z1.sum(axis=0)['Reward']
        rew_confidence_bound = math.sqrt(-(math.log(delta/2)/(2*n)))

        rew_lower_bound = (arm_reward / z1.shape[0]) - rew_confidence_bound
        G = p_lower_bound * rew_lower_bound
        profit_per_arm.append(G)
    return np.max(profit_per_arm)

def get_learner(age,gender,learners,split_age=False,split_gender=False):
    if (split_gender):
        if(gender == 0):
            return learners[3]
        else: return learners[4]
    if (split_age):
        if(age == 0):
            return learners[1]
        else:return learners[2]
    else:return learners[0]


def generate_reward(age,gender,probabilities,pulled_arm):
    if (age == 0 and gender == 0):
        reward = np.random.binomial(1, probabilities[0][pulled_arm])
        return reward
    if(age == 0 and gender == 1):
        reward = np.random.binomial(1, probabilities[1][pulled_arm])
        return reward
    else:
        reward = np.random.binomial(1, probabilities[2][pulled_arm])
        return reward




for e in range(0,n_experiments):
    learners = []
    z = pd.DataFrame(columns=['Gender', 'Age', 'Arm', 'Reward'])
    print('Experiment {}'.format(e))
    env = Environment(n_arms=n_arms, probabilities=p)
    ucb_learner1 = UCB1_Learner(n_arms=n_arms)
    ucb_learner2 = UCB1_Learner(n_arms=n_arms)
    learners.append(ucb_learner1)
    check_split_on_age = True
    check_split_on_gender = False
    split_age = False
    split_gender = False
    for i in range(0,T):
        gender = np.random.binomial(1, 0.5)
        age = np.random.binomial(1, 0.5)
        ucb_learner1 = get_learner(age, gender, learners, split_age, split_gender)
        if ((i % 7) == 0 and i != 0):
            if (check_split_on_age):
                split_age = split_on_age(z)
                if (split_age):
                    #print('split on age')
                    learner_under = ucb_learner1
                    learner_over = ucb_learner1
                    learners.append(learner_under)
                    learners.append(learner_over)
                    check_split_on_age = False
                    check_split_on_gender = True


            if(check_split_on_gender):
                split_gender = split_on_gender(z)
                if (split_gender):
                    learner_under_men = ucb_learner1
                    learner_under_women = ucb_learner1
                    learners.append(learner_under_men)
                    learners.append(learner_under_women)
                    check_split_on_gender = False

        #Thomposon Sampling Learner
        pulled_arm = ucb_learner1.pull_arm()
        reward = generate_reward(age,gender,p,pulled_arm)
        ucb_learner1.update(pulled_arm, reward)

        z = z.append({'Gender' : gender, 'Age': age,'Arm':pulled_arm,'Reward':reward},ignore_index=True)

        #Greedy Learner
        pulled_arm = ucb_learner2.pull_arm()
        reward = generate_reward(age,gender,p,pulled_arm)
        ucb_learner2.update(pulled_arm, reward)

    ts_rewards_per_experiment.append(ucb_learner1.collected_rewards)
    gr_rewards_per_experiment.append(ucb_learner2.collected_rewards)


plt.figure(0)
plt.xlabel('t')
plt.ylabel('Reward')
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'g')
plt.plot(np.mean(gr_rewards_per_experiment, axis=0), 'y')
plt.plot(T * [opt], '--k')
plt.legend(['Contextual-UCB1','UCB1','Optimum'])
plt.show()


plt.figure(0)
plt.xlabel('T')
plt.ylabel('Regret')
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment,axis=0)),'g')
plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment,axis=0)),'y')
plt.legend(['Contextual-UCB1','UCB1'])
plt.show()
