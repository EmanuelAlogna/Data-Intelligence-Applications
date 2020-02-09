import numpy as np
from SocialNetwork import *
import time
import matplotlib.pyplot as plt
from Environment import *
from LinearUCB import *

np.random.seed(123)
budget = 200
T = 50     # time horizon
c = 2   # exploration parameter
print("Creating graph and assigning probabilities ...")
ntwk = SocialNetwork(n_nodes=100, max_conn_per_node=10, budget=budget)
ntwk.assign_probabilities(feature=True)
ntwk.assign_costs()
g = ntwk.get_graph()
n_edges = len(ntwk.graph.edges)
print("Graph with {} edges and {} nodes created ".format(n_edges, ntwk.n_nodes))
probabilities = ntwk.get_probabilities()
print('True probabilities: {}'.format(probabilities))
print()
learner = LinearUCB(ntwk, 4, 1)
env = Environment(ntwk)

opt_seeds = celf(ntwk, budget, probabilities)
opt = information_cascade(ntwk, opt_seeds, probabilities, mc=100)
print('Optimal spread: {}'.format(opt))
print("Optimal seeds: {}".format(opt_seeds))
print()
spreads = []

for t in range(T):
    start_time = time.time()
    pulled_arm = learner.pull_superarm()
    reward = env.round(pulled_arm)
    learner.update(pulled_arm, reward)
    seeds = celf(ntwk, budget, np.clip(learner.get_estimated_probabilities(), 0, 1))
    spreads.append(information_cascade(ntwk, seeds, np.clip(learner.get_estimated_probabilities(), 0, 1), mc=100))
    print('Time for iteration {} : {}'.format(t, time.time() - start_time))

print('Optimal spread {}'.format(opt))
print(spreads)
print('Opt-spread: {}'.format(opt - spreads))
print(np.cumsum(np.abs((opt - spreads))))
plt.plot(np.cumsum(np.abs((opt - spreads))))
plt.show()
