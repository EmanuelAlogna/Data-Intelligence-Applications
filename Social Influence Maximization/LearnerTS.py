import numpy as np
from SocialNetwork import *
import time
import matplotlib.pyplot as plt
from Environment import *


class LearnerTS:
    def __init__(self,network):
        self.network = network
        self.graph = network.get_graph()
        self.alfa = dict.fromkeys(self.network.get_graph().edges, 1)
        self.beta = dict.fromkeys(self.network.get_graph().edges, 1)

    def pull_superarm(self):
        budget = self.network.get_budget()
        graph = self.network.get_graph()
        superarm = set()
        prob = np.random.beta(list(self.alfa.values()), list(self.beta.values()))

        seeds = celf(self.network, budget, prob)
        for seed in seeds:
            for u,v in graph.out_edges(seed):
                superarm.add((u, v))
        return superarm

    def update(self,pulled_arm,reward):
        graph = self.network.get_graph()
        seeds = set(u for (u, v) in pulled_arm)
        reachable_nodes = set(seeds)
        # Build a graph containing only the activated edges that are reachable from at least one seed
        live_edge_g = nx.DiGraph()
        live_edge_g.add_edges_from(reward)
        live_edge_g.add_nodes_from(seeds)

        for seed in seeds:
            reachable_nodes = reachable_nodes.union(nx.descendants(live_edge_g, seed))

        unreachable_nodes = set(graph.nodes).difference(reachable_nodes)
        live_edge_g.remove_nodes_from(unreachable_nodes)

        update = 0
        for node in live_edge_g.nodes:
            for u, v in graph.out_edges(node):
                if ((u, v) in live_edge_g.out_edges(node)):
                    self.alfa[(u, v)] += 1
                else:
                    self.beta[(u, v)] += 1

    def get_estimated_probabilities(self):
        prob = np.random.beta(list(self.alfa.values()), list(self.beta.values()))
        return prob