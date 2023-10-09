from typing import Type, Union
from agents.base_agent import BaseAgent
from agents.SimpleAgent import SimpleAgent
from simulation.AgentFactory import AgentFactory


class AgentManager:
    def __init__(
        self,
        num_agents: int,
        agent_class: Type[Union[BaseAgent, SimpleAgent]] = BaseAgent,
    ):
        self.agents = self.initialize_agents(num_agents, agent_class)

    @staticmethod
    def initialize_agents(num_agents, agent_class):
        agents = []
        if isinstance(agent_class, BaseAgent):
            return [agent_class(agent_id=i) for i in range(num_agents)]
        else:
            for i in range(num_agents):
                agents.append(AgentFactory.create_random_agent(agent_id=i))
        return agents
