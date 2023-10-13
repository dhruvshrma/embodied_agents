import random
from typing import List, Union, Any
from pydantic import BaseModel, Field, validator, constr
from agents.SimpleAgent import SimpleAgent, MediatingAgent
from agents.base_agent import BaseAgent
from interactions.DialogueSimulation import DialogueSimulator
from interactions.VoterModel import VoterModel
from simulation.AgentFactory import AgentFactory
from simulation.AgentManager import AgentManager
import networkx as nx

from environments.GraphEnvironment import GraphEnvironment
from configs.configs import (
    GraphEnvironmentConfig,
    TopologyType,
    LLMConfig,
    ModelType,
    SimulationConfig,
)
from utils.event_handler import EventHandler, AgentSpoke


def random_selector(agents: List[SimpleAgent]) -> int:
    """A simple selector function that chooses an agent at random."""
    return random.choice(range(len(agents)))


class SimulationRunner(BaseModel):
    config: SimulationConfig
    interaction_model: Union[DialogueSimulator, VoterModel]

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def attach_agents_to_nodes(
        graph: nx.Graph, agents: List[Union[BaseAgent, SimpleAgent]]
    ):
        for i, agent in enumerate(agents):
            graph.nodes[i]["agent"] = agent

    @validator("interaction_model", pre=True, always=True)
    def validate_interaction_model(cls, interaction_model, values):
        config = values.get("config")
        # Initialize the graph environment
        env_config = GraphEnvironmentConfig(
            num_agents=config.num_agents,
            topology=config.topology,
            small_world_k=config.small_world_k,
            small_world_p=config.small_world_p,
        )
        env = GraphEnvironment(config=env_config)

        agent_manager = AgentManager(
            num_agents=config.num_agents, agent_class=SimpleAgent
        )

        llm_config = LLMConfig(
            model_type=config.model_type, temperature=config.temperature
        )

        # Initialize the chat model
        # if (
        #     config.model_type == ModelType.GPT3
        #     or config.model_type == ModelType.GPT3BIS
        # ):
        #     from langchain.chat_models import ChatOpenAI
        #
        #     llm = ChatOpenAI(model=config.model_type, temperature=config.temperature)
        # else:
        #     from langchain.chat_models import ChatOllama
        #
        #     llm = ChatOllama(model=config.model_type, temperature=config.temperature)

        # Set the model for the agents
        for agent in agent_manager.agents:
            agent.set_model(llm_config)
            agent.create_agent_description()
            agent.create_system_message(topic=config.topic)

        mediating_agent = AgentFactory.create_mediating_agent(topic=config.topic)
        mediating_agent.set_model(llm_config)
        mediating_agent.set_system_message()

        simulator = interaction_model(
            environment=env,
            mediating_agent=mediating_agent,
            agents=agent_manager.agents,
            selection_function=random_selector,
            topic=config.topic,
        )
        cls.attach_agents_to_nodes(simulator.environment.graph, simulator.agents)
        return simulator

    def run_simulation(self):
        for i in range(self.config.num_rounds):
            EventHandler.handle(
                AgentSpoke(agent_name="SYSTEM", message=f"----\nRound {i+1}")
            )
            name, message = self.interaction_model.step()
            EventHandler.handle(AgentSpoke(agent_name=name, message=message))
            EventHandler.handle(AgentSpoke(agent_name="SYSTEM", message="----"))
        EventHandler.handle(
            AgentSpoke(agent_name="SYSTEM", message="Simulation complete")
        )
