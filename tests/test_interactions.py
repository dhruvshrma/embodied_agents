import pytest

from agents.base_agent import BaseAgent
from interactions.VoterModel import VoterModel
from interactions.DialogueSimulation import DialogueSimulator
from environments.GraphEnvironment import GraphEnvironment, GraphEnvironmentConfig
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


def random_selector(step: int, agents: List[SimpleAgent], env: GraphEnvironment) -> int:
    """A simple selector function that chooses an agent at random."""
    return random.choice(range(len(agents)))


@pytest.fixture
def setup_graph_environment():
    config = GraphEnvironmentConfig(
        num_agents=3, topology="star", small_world_k=2, small_world_p=0.3
    )
    env = GraphEnvironment(agent_class=SimpleAgent, config=config)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    for agent in env.agents:
        agent.model = llm
    sim = DialogueSimulator(environment=env, selection_function=random_selector)

    return env, sim


def test_initialization(setup_graph_environment):
    env, sim = setup_graph_environment
    assert sim._step == 0
    assert len(sim.agents) == env.config.num_agents
    assert sim.select_next_speaker == random_selector
    assert len(sim.history) == 0


def test_inject(setup_graph_environment):
    env, sim = setup_graph_environment

    sim.inject("TestUser", "Hello agents!")
    for agent in sim.agents:
        assert agent.message_history[-1] == "TestUser: Hello agents!"


def test_step(setup_graph_environment):
    env, sim = setup_graph_environment

    name, message = sim.step()
    assert name in [agent.name for agent in sim.agents]
    assert message is not None


def test_reset(setup_graph_environment):
    env, sim = setup_graph_environment

    sim.inject("TestUser", "Hello agents!")
    sim.step()
    sim.reset()

    assert sim._step == 0
    assert all([len(agent.message_history) == 1 for agent in sim.agents])
    for agent in sim.agents:
        assert agent.message_history == ["Here is the conversation so far."]
    assert len(sim.history) == 0
