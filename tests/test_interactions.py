import pytest

from agents.base_agent import BaseAgent
from interactions.VoterModel import VoterModel
from interactions.DialogueSimulation import DialogueSimulator
from environments.GraphEnvironment import GraphEnvironment, GraphEnvironmentConfig
from configs.configs import ModelType, LLMConfig
from simulation.AgentFactory import AgentFactory
from agents.SimpleAgent import SimpleAgent
from langchain.chat_models import ChatOpenAI
from typing import List
import random
from dotenv import load_dotenv

load_dotenv()


def test_voter_model_interaction():
    agent1 = BaseAgent(agent_id=1)
    agent2 = BaseAgent(agent_id=2)
    agent1.set_opinion(1)
    agent2.set_opinion(-1)

    vm = VoterModel()
    vm.interact(agent1, [agent2])
    assert agent1.get_opinion() == -1


def random_selector(agents: List[SimpleAgent]) -> int:
    """A simple selector function that chooses an agent at random."""
    return random.choice(range(len(agents)))


@pytest.fixture
def setup_graph_environment():
    config = GraphEnvironmentConfig(
        num_agents=3, topology="star", small_world_k=2, small_world_p=0.3
    )
    llm_config = LLMConfig(model_type=ModelType.GPT3, temperature=0.0)
    env = GraphEnvironment(config=config)
    agents= [AgentFactory.create_testuser_agent(agent_id=0)]
    for i in range(1, config.num_agents):
        agents.append(AgentFactory.create_random_agent(agent_id=i))
    for agent in agents:
        agent.set_model(llm_config)
    sim = DialogueSimulator(environment=env, selection_function=random_selector,
                            agents=agents) 
    
    for i, agent in enumerate(agents):
            env.graph.nodes[i]["agent"] = agent
                            
    return env, sim 


def test_initialization(setup_graph_environment):
    env, sim  = setup_graph_environment
    assert sim.environment == env
    assert sim.select_next_speaker == random_selector
    assert len(sim.history) == 0
    assert sim.agents[0].name == "TestUser"
    assert sim.agents[0].agent_id == 0
    
    for agent in sim.agents:
        assert agent.model is not None
        assert isinstance(agent.model, ChatOpenAI)

def test_inject(setup_graph_environment):
    env, sim = setup_graph_environment

    sim.inject(idx=0, message="Hello agents!")
    for agent in sim.agents:
        assert agent.message_history[-1] == "TestUser: Hello agents!"


def test_step(setup_graph_environment):
    env, sim = setup_graph_environment

    name, message = sim.step()
    assert name in [agent.name for agent in sim.agents]
    assert message is not None


def test_reset(setup_graph_environment):
    env, sim = setup_graph_environment

    sim.inject(idx=0, message = "Hello agents!")
    name, message = sim.step()
    sim.reset()

    assert sim._step == 0
    assert all([len(agent.message_history) == 1 for agent in sim.agents])
    for agent in sim.agents:
        assert agent.message_history == ["Here is the conversation so far."]
    assert len(sim.history) == 0
