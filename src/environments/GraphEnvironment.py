import random
from typing import Type
from typing import Union

import networkx as nx
from faker import Faker
from pydantic import BaseModel, validator

from agents.SimpleAgent import SimpleAgent
from agents.base_agent import BaseAgent

fake = Faker()


class GraphEnvironmentConfig(BaseModel):
    topology: str
    num_agents: int
    small_world_k: int = 2
    small_world_p: float = 0.3
    scale_free_m: int = 1

    @validator("topology")
    def validate_topology(cls, value):
        valid_topologies = ["star", "small-world", "scale-free"]
        if value not in valid_topologies:
            raise ValueError(f"Invalid topology. Choose from {valid_topologies}")
        return value

    @validator("num_agents", pre=True)
    def validate_num_agents(cls, value):
        if value <= 1:
            raise ValueError("num_agents must be greater than 1")
        return value

    @validator("small_world_k", pre=True, always=True)
    def validate_small_world_k(cls, k, values):
        topology = values.get("topology")
        num_agents = values.get("num_agents")

        if topology == "small-world":
            if num_agents is not None:  # Ensure num_agents has been processed
                if num_agents <= k:
                    raise ValueError("num_agents must be greater than small_world_k")
                elif k < 2:
                    raise ValueError("small_world_k must be greater than 1")
                elif num_agents == 2 and k == 1:
                    raise ValueError("For num_agents=2, small_world_k cannot be 1")
        return k

    @validator("small_world_p")
    def validate_small_world_p(cls, value):
        if not (0 <= value <= 1):
            raise ValueError("small_world_p must be between 0 and 1")
        return value

    @validator("scale_free_m")
    def validate_scale_free_m(cls, value):
        if value <= 0:
            raise ValueError("scale_free_m must be greater than 0")
        return value


class GraphEnvironment:
    def __init__(
        self,
        config: GraphEnvironmentConfig,
        agent_class: Type[Union[BaseAgent, SimpleAgent]] = BaseAgent,
    ):
        self.config = config
        self.graph = self.create_topology()
        self.agents = self.initialize_agents(agent_class)
        self.attach_agents_to_nodes()

    def create_topology(self):
        if self.config.topology == "star":
            return nx.star_graph(self.config.num_agents - 1)
        elif self.config.topology == "small-world":
            return nx.watts_strogatz_graph(
                self.config.num_agents,
                k=self.config.small_world_k,
                p=self.config.small_world_p,
            )
        elif self.config.topology == "scale-free":
            return nx.barabasi_albert_graph(
                self.config.num_agents, m=self.config.scale_free_m
            )

    def initialize_agents(self, agent_class):
        if isinstance(agent_class, BaseAgent):
            return [agent_class(agent_id=i) for i in range(self.config.num_agents)]
        else:
            return [
                agent_class(agent_id=i, name=fake.name())
                for i in range(self.config.num_agents)
            ]

    def attach_agents_to_nodes(self):
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
