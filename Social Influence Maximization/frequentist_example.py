from Environment import *
import matplotlib.pyplot as plt
from random import sample
from SocialNetwork import *
import time
from FrequentistLearnerUCB import *

np.random.seed(123)
b = 200
net = SocialNetwork(n_nodes=100, max_conn_per_node=7, budget=b)
net.assign_probabilities()
net.assign_costs()
g = net.get_graph()
T = 200

print("Creating graph and assigning probabilities ...")
n_edges = len(net.graph.edges)
print("Graph with {} edges and {} nodes created ".format(n_edges, net.n_nodes))
probabilities = net.get_probabilities()
print('True probabilities: {}'.format(probabilities))
print()

env = Environment(net)
learner = FrequentistLearnerUCB(net)

opt_seeds = celf(net, b, probabilities)
opt = information_cascade(net, opt_seeds, probabilities, mc=100)
print("Optimal spread: ", opt)
print("Optimal seeds ", opt_seeds)
print()
spreads = []


for t in range(T):
    start_time = time.time()
    # select seeds according to estimated probabilities. At round 1 the empirical mean values are initialized to 1.
    superarm = learner.pull_superarm()
    # now I run IC according to the true probabilities, using seeds coming form the pulled_arms
    # what is returned is the set of nodes that were activated during the IC

    reward = env.round(superarm)
    learner.update(superarm, reward)
    seeds = celf(net, b, list(learner.get_estimated_probabilities().values()))
    p = list(learner.get_estimated_probabilities().values())
    spreads.append(information_cascade(net, seeds, p, mc=100))
    print('Time for round {} : {}'.format(t, time.time() - start_time))

print(spreads)

print('True probabilities: {}'.format(probabilities))

print('Estimated probabilities: {}'.format(list(learner.get_estimated_probabilities().values())))

plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("T")
plt.plot(np.cumsum(np.abs(opt - spreads)), 'b')
plt.legend(["UCB"])
plt.show()
