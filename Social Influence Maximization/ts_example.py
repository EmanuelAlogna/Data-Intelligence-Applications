import numpy as np
from SocialNetwork import *
import time
import matplotlib.pyplot as plt
from Environment import *
from LearnerTS import *

np.random.seed(123)
budget = 200
T = 200     # time horizon
print("Creating graph and assigning probabilities ...")
ntwk = SocialNetwork(n_nodes=100, max_conn_per_node=15, budget=budget)
ntwk.assign_probabilities()
ntwk.assign_costs()
g = ntwk.get_graph()
n_edges = len(ntwk.graph.edges)
print("Graph with {} edges and {} nodes created ".format(n_edges, ntwk.n_nodes))
probabilities = ntwk.get_probabilities()
print('True probabilities: {}'.format(probabilities))
print()
ts_learner = LearnerTS(ntwk)
env = Environment(ntwk)

opt_seeds = celf(ntwk, budget, probabilities)
opt = information_cascade(ntwk, opt_seeds, probabilities, mc=100)
print("Optimal spread: {}".format(opt))
print("Optimal seeds: {}".format(opt_seeds))
print()
spreads = []

for t in range(T):
    start_time = time.time()
    seeds = set()
    pulled_arm = ts_learner.pull_superarm()
    reward = env.round(pulled_arm)
    ts_learner.update(pulled_arm, reward)
    estimated_seeds = celf(ntwk, budget, list(ts_learner.get_estimated_probabilities()))
    spreads.append(information_cascade(ntwk, estimated_seeds, list(ts_learner.get_estimated_probabilities()), mc=100))
    print('Time for iteration {} : {}'.format(t, time.time() - start_time))

print(spreads)
print('Opt-spread: {}'.format(opt - spreads))
print(np.cumsum(np.abs((opt - spreads))))
plt.plot(np.cumsum(np.abs((opt - spreads))))
plt.show()
