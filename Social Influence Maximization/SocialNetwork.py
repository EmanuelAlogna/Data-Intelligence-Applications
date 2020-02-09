import numpy as np
import networkx as nx


class SocialNetwork:
    def __init__(self, n_nodes, max_conn_per_node=15, n_features=4, budget=500):
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.budget = budget
        self.max_conn_per_node = max_conn_per_node
        self.graph = self.build_graph()

    def build_graph(self):
        while True:
            deg_list = np.random.randint(low=1, high=self.max_conn_per_node, size=self.n_nodes)
            if (nx.is_graphical(deg_list)):
                break
        graph = nx.random_degree_sequence_graph(deg_list, seed=123, tries=1000)
        graph = graph.to_directed()
        return graph

    def assign_probabilities(self, feature=False):

        # Probabability is drawn uniformly in the range [0.0005,0.2]
        for edge in self.graph.edges:
            self.graph[edge[0]][edge[1]]['weight'] = np.random.uniform(low=0.0005, high=0.5)

        if feature:
            self.theta = np.random.dirichlet(np.ones(self.n_features), size=1)
            print('True theta: {}'.format(self.theta))
            for edge in self.graph.edges:
                features = np.random.uniform(low=0, high=0.5, size=self.n_features)
                features = features.reshape(4, 1)
                self.graph[edge[0]][edge[1]]['features'] = features
                self.graph[edge[0]][edge[1]]['weight'] = np.dot(self.theta, features).item()

    def assign_costs(self):
        # Il costo di un nodo dipende dalla media dei pesi associati al archi uscenti da un nodo
        for node in self.graph.nodes:
            weights = []
            node_deg = self.graph.out_degree[node]
            for i in [x for x in self.graph.neighbors(node)]:
                weights.append(self.graph[node][i]['weight'])
            cost = 50 + int(10 * np.mean(weights))
            self.graph.nodes[node]['cost'] = cost

    def get_graph(self):
        return self.graph

    def get_number_of_nodes(self):
        return self.n_nodes

    def get_probabilities(self):
        p = []
        for node1, node2, data in self.graph.edges(data=True):
            p.append(data['weight'])
        return p

    def get_budget(self):
        return self.budget


def information_cascade(network, seeds, p, mc=100):
    """
     Input:  network object, set of seed nodes, edge probability
             and the number of Monte-Carlo simulations
     Output: average number of nodes influenced by the seed nodes
    """

    graph = network.get_graph()
    spread = []
    # Run multiple MC simulations
    for i in range(mc):
        success = np.random.binomial(1, p)
        active_edges = [edge for i, edge in enumerate(graph.edges) if success[i] == 1]
        reachable_nodes = set(seeds)
        # Build a graph containing only the activated edges that are reachable from at least one seed

        live_edge_g = nx.DiGraph()
        live_edge_g.add_edges_from(active_edges)
        live_edge_g.add_nodes_from(seeds)

        for seed in seeds:
            reachable_nodes = reachable_nodes.union(nx.descendants(live_edge_g, seed))

        unreachable_nodes = set(graph.nodes).difference(reachable_nodes)
        live_edge_g.remove_nodes_from(unreachable_nodes)

        spread.append(len(live_edge_g.nodes))

    return np.mean(spread)


def greedy(network, budget, p, delta=0.95):
    """
    Input:  network object, budget, edge probability
            and the number of Monte-Carlo simulations
    Output: set of seeds that maximize the influence
    """
    seeds = []
    residual_budget = budget
    g = network.get_graph()
    nodes = list(g.nodes)
    epsilon = 0.1
    spread = 0
    while residual_budget >= 0:

        best_spread = 0
        seed_cost = 1
        for node in (set(nodes) - set(seeds)):

            n_simulations = int((1 / (epsilon ** 2)) * np.log(len(seeds + [node]) + 1) * np.log(1 / delta))
            cost = g.nodes[node]['cost']
            new_spread = (information_cascade(network,seeds + [node], p, n_simulations) - spread)/cost

            if new_spread > best_spread:
                best_spread = spread
                new_seed = node
                seed_cost = cost

        seeds.append(new_seed)
        residual_budget = residual_budget - seed_cost

        n_simulations = int((1 / (epsilon ** 2)) * np.log(len(seeds) + 1) * np.log(1 / delta))
        spread = information_cascade(network, seeds, p, n_simulations)

    if residual_budget < 0:
        seeds.pop()
        residual_budget += seed_cost
    return seeds


def celf(network, budget, p, delta=0.95):
    """
    Input:  network object, budget, edge probability
            and the number of Monte-Carlo simulations
    Output: set of seeds that maximize the influence
    """
    seeds = []
    residual_budget = budget
    g = network.get_graph()
    nodes = list(g.nodes)
    epsilon = 0.1
    marginal_increase = dict.fromkeys(g.nodes,0)
    nodes_to_evaluate = set(marginal_increase.keys())

    for node in nodes_to_evaluate:

        n_simulations = int((1 / (epsilon ** 2)) * np.log(len(seeds + [node]) + 1) * np.log(1 / delta))
        cost = g.nodes[node]['cost']
        spread = information_cascade(network, seeds + [node], p, n_simulations) / cost
        marginal_increase[node] = spread

    while residual_budget >= 0:

        new_seed = max(marginal_increase, key=marginal_increase.get)
        seed_cost = g.nodes[new_seed]['cost']

        n_simulations = int((1 / (epsilon ** 2)) * np.log(len(seeds + [new_seed]) + 1) * np.log(1 / delta))

        spread =  information_cascade(network, seeds, p, n_simulations)

        new_spread = information_cascade(network, seeds + [new_seed], p, n_simulations)

        marginal_increase[new_seed] = (new_spread - spread) / seed_cost

        if max(marginal_increase, key=marginal_increase.get) == new_seed:
            seeds.append(new_seed)
            marginal_increase.pop(new_seed)
            residual_budget = residual_budget - seed_cost

    if residual_budget < 0:
        seeds.pop()
        residual_budget += seed_cost
    return seeds
