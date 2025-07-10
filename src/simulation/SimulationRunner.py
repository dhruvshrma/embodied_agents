import random
from typing import List, Union
from pydantic import BaseModel, field_validator, ConfigDict
from agents.SimpleAgent import SimpleAgent
from agents.base_agent import BaseAgent
from interactions.DialogueSimulation import DialogueSimulator
from interactions.VoterModel import VoterModel
from simulation.AgentFactory import AgentFactory
from simulation.AgentManager import AgentManager
import networkx as nx

from environments.GraphEnvironment import GraphEnvironment
from configs.configs import (
    GraphEnvironmentConfig,
    LLMConfig,
    SimulationConfig,
)
from utils.event_handler import EventHandler, AgentSpoke
from opinion_dynamics.OpinionAnalyzer import OpinionAnalyzer


def random_selector(agents: List[SimpleAgent]) -> int:
    """A simple selector function that chooses an agent at random."""
    return random.choice(range(len(agents)))


class SimulationRunner(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    config: SimulationConfig
    interaction_model: Union[DialogueSimulator, VoterModel]

    @staticmethod
    def attach_agents_to_nodes(
        graph: nx.Graph, agents: List[Union[BaseAgent, SimpleAgent]]
    ):
        for i, agent in enumerate(agents):
            graph.nodes[i]["agent"] = agent

    @field_validator("interaction_model", mode='before')
    @classmethod
    def validate_interaction_model(cls, interaction_model, info):
        values = info.data if info.data else {}
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

        # Create OpinionAnalyzer for tracking opinion dynamics
        opinion_analyzer = OpinionAnalyzer(
            llm_client=None,  # Will be set based on model type
            update_frequency=config.opinion_update_frequency
        )
        
        # Set the appropriate LLM client for opinion analysis
        if config.model_type.value in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"]:
            from src.llm.openai_client import OpenAIClient
            opinion_analyzer.llm_client = OpenAIClient(
                model=config.model_type.value,
                temperature=0.3  # Lower temperature for more consistent opinion analysis
            )
        else:
            from src.llm.ollama_client import OllamaClient
            opinion_analyzer.llm_client = OllamaClient(
                model=config.model_type.value,
                temperature=0.3
            )

        # Initialize agent opinions from personas
        for agent in agent_manager.agents:
            initial_opinion = opinion_analyzer.initialize_opinion_from_persona(agent)
            agent.set_opinion(initial_opinion)

        simulator = interaction_model(
            environment=env,
            mediating_agent=mediating_agent,
            agents=agent_manager.agents,
            selection_function=random_selector,
            topic=config.topic,
            opinion_analyzer=opinion_analyzer,
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
