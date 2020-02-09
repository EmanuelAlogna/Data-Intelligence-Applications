import numpy as np
from LearnerUCB import *
from Environment import *
import matplotlib.pyplot as plt
import time

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

env = Environment(ntwk)
learner = LearnerUCB(ntwk)
print('True probabilities: {}'.format(probabilities))
print()

opt_seeds = celf(ntwk, budget, probabilities)
opt = information_cascade(ntwk, opt_seeds, probabilities, mc=100)
print('Optimal spread: {}'.format(opt))
print("Optimal seeds: {}".format(opt_seeds))
print()
spreads = []

for t in range(T):
    start_time = time.time()
    seeds = set()
    pulled_arm = learner.pull_superarm()
    reward = env.round(pulled_arm)
    learner.update(pulled_arm, reward)
    estimated_seeds = celf(ntwk, budget, list(learner.get_estimated_probabilities().values()))
    spreads.append(information_cascade(ntwk, estimated_seeds, list(learner.get_estimated_probabilities().values()), mc=100))
    print('Time for iteration {} : {}'.format(t, time.time()-start_time))
print(spreads)
print('Opt-spread: {}'.format(opt - spreads))
print(np.cumsum(np.abs((opt - spreads))))
plt.plot(np.cumsum(np.abs((opt - spreads))))
plt.show()

print('True probabilities: {}'.format(probabilities))
print('Estimated probabilities: {}'.format(list(learner.get_estimated_probabilities().values())))

