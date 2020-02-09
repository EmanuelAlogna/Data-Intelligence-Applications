from SocialNetwork import *


class LearnerUCB:
    def __init__(self,network):
        self.network = network
        self.empirical_mean = dict.fromkeys(self.network.get_graph().edges,1)
        self.empirical_mean_no_bound = dict.fromkeys(self.network.get_graph().edges, 1)
        self.cumulative_reward = dict.fromkeys(self.network.get_graph().edges,0)
        self.T = dict.fromkeys(self.network.get_graph().edges,0)
        self.t = 0

    def pull_superarm(self):
        budget = self.network.get_budget()
        graph = self.network.get_graph()
        superarm = set()
        seeds = celf(self.network,budget,list(self.empirical_mean.values()))
        for seed in seeds:
            for u,v in graph.out_edges(seed):
                superarm.add((u,v))
        return superarm

    def update(self, pulled_arm, reward):
        self.t += 1
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
                self.T[(u, v)] += 1
                if ((u, v) in live_edge_g.out_edges(node)):
                    self.cumulative_reward[(u, v)] += 1
                bound = np.sqrt((3*np.log(self.t))/(2 * self.T[(u,v)]))
                self.empirical_mean_no_bound[(u,v)] = min(self.cumulative_reward[(u, v)] / self.T[(u, v)] ,1)
                self.empirical_mean[(u, v)] = min(self.cumulative_reward[(u, v)] / self.T[(u, v)] + bound,1)

    def get_estimated_probabilities(self):
        return self.empirical_mean_no_bound

