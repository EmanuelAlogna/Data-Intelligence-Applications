import numpy as np
from SocialNetwork import *


class Environment:
    def __init__(self, network):
        self.network = network

    def round(self, pulled_arm):
        success = np.random.binomial(1, self.network.get_probabilities())
        active_edges = [edge for i, edge in enumerate(self.network.get_graph().edges) if success[i] == 1]
        return active_edges
