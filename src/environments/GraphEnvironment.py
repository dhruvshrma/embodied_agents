import networkx as nx
import random
from agents.base_agent import BaseAgent


class GraphEnvironment:
    def __init__(self, agent_class=BaseAgent, topology="star", num_agents=5):
        if topology == "star":
            self.graph = nx.star_graph(num_agents - 1)  # One node is central
        elif topology == "small-world":
            self.graph = nx.watts_strogatz_graph(num_agents, k=2, p=0.3)
        elif topology == "scale-free":
            self.graph = nx.barabasi_albert_graph(num_agents, 1)

        self.agents = [agent_class(agent_id=i) for i in range(num_agents)]
        for i, agent in enumerate(self.agents):
            self.graph.nodes[i]["agent"] = agent

    def initialize_opinions_randomly(self):
        for agent in self.agents:
            agent.set_opinion(random.choice([-1, 0, 1]))

    def get_neighbors(self, agent):
        agent_node = [
            node for node, data in self.graph.nodes(data=True) if data["agent"] == agent
        ][0]
        neighbor_nodes = list(self.graph[agent_node])
        return [self.graph.nodes[node]["agent"] for node in neighbor_nodes]
