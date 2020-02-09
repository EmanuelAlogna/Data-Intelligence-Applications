import matplotlib.pyplot as plt
import time
from Environment import *

T = 200     # time horizon
budget = 400
print("Creating graph and assigning probabilities ...")
ntwk = SocialNetwork(n_nodes=1000, max_conn_per_node=15)
ntwk.assign_probabilities()
ntwk.assign_costs()
g = ntwk.get_graph()
n_edges = len(ntwk.graph.edges)
print("Graph with {} edges and {} nodes created ".format(n_edges, ntwk.n_nodes))
probabilities = ntwk.get_probabilities()
print('Probabilities: {}'.format(probabilities))
print()

start_time = time.time()
deltas = [0.95, 0.8, 0.4, 0.2]
spreads = []
for delta in deltas:
    start_time = time.time()
    print('Simulation for delta: {}'.format(delta))
    seeds = greedy(ntwk, budget, probabilities, delta=delta)
    spread = information_cascade(ntwk, seeds, probabilities, 1000)
    spreads.append(spread)
    print('Seeds: {}'.format(seeds))
    print('Spread: {}'.format(spread))
    print('Total time: {}'.format(time.time() - start_time))
    print()

plt.plot(deltas, spreads)
plt.show()
