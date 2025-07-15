import pytest
from utils.log_config import setup_logging
from personas.Persona import Persona
from agents.SimpleAgent import SimpleAgent, MediatingAgent
from simulation.AgentManager import AgentManager
from interactions.DialogueSimulation import DialogueSimulator
from environments.GraphEnvironment import GraphEnvironment
from configs.configs import LLMConfig, ModelType, GraphEnvironmentConfig
from llm.openai_client import OpenAIClient
from dotenv import load_dotenv

load_dotenv()
logger = setup_logging()


@pytest.fixture
def language_model():
    llm = OpenAIClient(model="gpt-3.5-turbo", temperature=0.0)
    return llm


@pytest.fixture
def llm_config():
    config = LLMConfig(
        model_type=ModelType.GPT3,
        temperature=0.0,
        presence_penalty=1.0,
        frequency_penalty=1.0,
    )
    return config


def random_selector(agents):
    return 0


@pytest.fixture
def setup_graph_environment(llm_config):
    from simulation.SimulationRunner import SimulationRunner
    env_config = GraphEnvironmentConfig(
        num_agents=3, topology="star", small_world_k=2, small_world_p=0.3
    )
    env = GraphEnvironment(config=env_config)
    agent_manager = AgentManager(num_agents=3, agent_class=SimpleAgent)
    mediating_agent = MediatingAgent(
        name="Mediator", topic="A discussion on ice-cream flavors", agent_id=-1
    )
    for agent in agent_manager.agents:
        agent.set_model(llm_config)
        agent.create_agent_description()
        agent.create_system_message(topic="A discussion on ice-cream flavors")
    mediating_agent.set_model(llm_config)
    mediating_agent.set_system_message()
    simulator = DialogueSimulator(
        environment=env,
        mediating_agent=mediating_agent,
        agents=agent_manager.agents,
        selection_function=random_selector,
        topic="A discussion on ice-cream flavors",
    )
    # Attach agents to nodes like SimulationRunner does
    SimulationRunner.attach_agents_to_nodes(simulator.environment.graph, simulator.agents)
    return simulator


def test_initialization(setup_graph_environment):
    simulator = setup_graph_environment
    assert len(simulator.agents) == 3
    for agent in simulator.agents:
        assert isinstance(agent.model, OpenAIClient)
    assert simulator.topic == "A discussion on ice-cream flavors"
    assert simulator._step == 0
    assert len(simulator.history) == 0
    assert simulator.select_next_speaker == random_selector


def test_voter_model_interaction():
    """Test VoterModel interaction between agents."""
    from agents.base_agent import BaseAgent
    from interactions.VoterModel import VoterModel
    
    agent1 = BaseAgent(agent_id=1)
    agent2 = BaseAgent(agent_id=2)
    agent1.set_opinion(1)
    agent2.set_opinion(-1)

    vm = VoterModel()
    vm.interact(agent1, [agent2])
    assert agent1.get_opinion() == -1


def test_inject(setup_graph_environment):
    """Test injecting messages into the simulation."""
    simulator = setup_graph_environment
    
    # Test injecting from a specific agent (using integer index)
    simulator.inject(0, "Hello agents!")
    for agent in simulator.agents:
        assert agent.message_history[-1] == f"{simulator.agents[0].name}: Hello agents!"


def test_step(setup_graph_environment):
    """Test stepping through the simulation."""
    simulator = setup_graph_environment
    
    name, message = simulator.step()
    assert name in [agent.name for agent in simulator.agents] or name == "Mediator"
    assert message is not None
    assert simulator._step == 1


def test_reset(setup_graph_environment):
    """Test resetting the simulation."""
    simulator = setup_graph_environment
    
    # Inject a message and step through
    simulator.inject(0, "Hello agents!")
    simulator.step()
    
    # Reset the simulation
    simulator.reset()
    
    assert simulator._step == 0
    assert all([len(agent.message_history) == 1 for agent in simulator.agents])
    for agent in simulator.agents:
        assert agent.message_history == ["Here is the conversation so far."]
    assert len(simulator.history) == 0


def test_multiple_steps(setup_graph_environment):
    """Test multiple steps in the simulation."""
    simulator = setup_graph_environment
    
    # Take multiple steps
    for i in range(3):
        name, message = simulator.step()
        assert name is not None
        assert message is not None
        assert simulator._step == i + 1
    
    # Check that history is being maintained
    assert len(simulator.history) == 3


def test_mediator_functionality(setup_graph_environment):
    """Test mediator agent functionality."""
    simulator = setup_graph_environment
    
    # Test mediator injection
    simulator.inject(None, "")
    
    # Should be able to step through with mediator
    name, message = simulator.step()
    assert name is not None
    assert message is not None
