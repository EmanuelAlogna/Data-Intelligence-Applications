import numpy as np
from SocialNetwork import *
import time
import matplotlib.pyplot as plt
from Environment import *


class LinearUCB:
    def __init__(self,network, n_features, c):
        self.network = network
        self.n_features = n_features
        self.M = np.identity(self.n_features)
        self.b = np.zeros(self.n_features)
        self.b = self.b.reshape(4, 1)
        self.c = c
        self.theta = []
        self.graph = network.get_graph()

    def pull_superarm(self):
        inv_M = np.linalg.inv(self.M)
        self.theta = np.dot(inv_M, self.b)
        probabilities = []
        for edge in self.graph.edges:
            feature = self.graph[edge[0]][edge[1]]['features']
            ucb = np.clip(np.dot(feature.T, self.theta) + self.c * np.sqrt(np.dot(feature.T, np.dot(inv_M, feature))), 0, 1)
            self.graph[edge[0]][edge[1]]['ucb'] = ucb
            probabilities.append(ucb)

        budget = self.network.get_budget()
        superarm = set()
        seeds = celf(self.network, budget, probabilities)
        for seed in seeds:
            for u, v in self.graph.out_edges(seed):
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

        for node in live_edge_g.nodes:
            for u, v in graph.out_edges(node):
                feat = graph[u][v]['features']
                self.M = self.M + np.dot(feat,feat.T)
                if (u, v) in live_edge_g.out_edges(node):
                    self.b = self.b + feat

    def get_estimated_probabilities(self):
        estimated_probabilities = []
        for edge in self.graph.edges:
            prob = np.dot(self.theta.T,self.graph[edge[0]][edge[1]]['features'])
            estimated_probabilities.append(prob)
        return np.array(estimated_probabilities).ravel()

