import random
from typing import List, Union
from pydantic import BaseModel, Field, validator, constr
from agents.SimpleAgent import SimpleAgent, MediatingAgent
from interactions.DialogueSimulation import DialogueSimulator
from interactions.VoterModel import VoterModel
from environments.GraphEnvironment import GraphEnvironment, GraphEnvironmentConfig
from enum import Enum


def random_selector(agents: List[SimpleAgent]) -> int:
    """A simple selector function that chooses an agent at random."""
    return random.choice(range(len(agents)))


class ModelType(str, Enum):
    MISTRAL = "mistral:latest"
    GPT3 = "gpt-3.5-turbo"
    LLAMA2 = "llama2:13b-chat"
    LLAMA2BIS = "llama2-uncensored"


class TopologyType(str, Enum):
    SMALL_WORLD = "small-world"
    STAR = "star"
    SCALE_FREE = "scale-free"


class SimulationConfig(BaseModel):
    num_agents: int
    topic: str
    num_rounds: int
    topology: TopologyType = TopologyType.SMALL_WORLD
    model_type: ModelType = ModelType.GPT3

    temperature: float = 1.0
    small_world_k: int = 4
    small_world_p: float = 0.3


class SimulationRunner(BaseModel):
    config: SimulationConfig
    interaction_model: DialogueSimulator

    class Config:
        arbitrary_types_allowed = True

    @validator("interaction_model", pre=True, always=True)
    def validate_setup(cls, interaction_model, values):
        config = values.get("config")

        # Initialize the graph environment
        env_config = GraphEnvironmentConfig(
            num_agents=config.num_agents,
            topology=config.topology,
            small_world_k=config.small_world_k,
            small_world_p=config.small_world_p,
        )
        env = GraphEnvironment(agent_class=SimpleAgent, config=env_config)

        # Initialize the chat model
        if config.model_type == ModelType.GPT3:
            from langchain.chat_models import ChatOpenAI

            llm = ChatOpenAI(model=config.model_type, temperature=config.temperature)
        else:
            from langchain.chat_models import ChatOllama

            llm = ChatOllama(model=config.model_type, temperature=config.temperature)

        # Set the model for the agents
        for agent in env.agents:
            agent.model = llm

        # Initialize the mediating agent
        mediating_agent = MediatingAgent(name="Mediator", agent_id=-1)
        mediating_agent.model = llm
        mediating_agent.set_system_message()

        # Initialize the DialogueSimulation class and agent descriptions
        simulator = interaction_model(
            environment=env,
            mediating_agent=mediating_agent,
            selection_function=random_selector,
            topic=config.topic,
        )

        for agent in simulator.agents:
            agent.create_agent_description()
            agent.create_system_message(topic=simulator.topic)

        return simulator

    def run_simulation(self):
        for i in range(self.config.num_rounds):
            print("----")
            print(f"Round {i}")
            name, message = self.interaction_model.step()
            print(f"{name}: {message}")
            print("----")
