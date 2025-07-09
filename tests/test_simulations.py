import pytest
from simulation.SimulationRunner import SimulationRunner
from configs.configs import SimulationConfig, ModelType, TopologyType
from langchain.chat_models import ChatOpenAI, ChatOllama
from interactions.DialogueSimulation import DialogueSimulator
from dotenv import load_dotenv

load_dotenv()
params = [
    (TopologyType.SCALE_FREE, 1.0),]
# (TopologyType.SMALL_WORLD, 0.5), (TopologyType.STAR, 0.0)]



@pytest.fixture(params=params, scope="function")
def setup_sim_config(request):
    topology = request.param[0] if request.param else TopologyType.SCALE_FREE
    temperature = request.param[1] if request.param else 1.0
    print(f"Running test with topology={topology} and temperature={temperature}")
    sim_config = SimulationConfig(
        num_agents=7,
        topic="A discussion on ice-cream flavors",
        num_rounds=4,
        topology=topology,
        model_type=ModelType.LLAMA2,
        temperature=temperature,
    )
    return sim_config


def test_initialization(setup_sim_config):
    config = setup_sim_config
    runner = SimulationRunner(config=config, interaction_model=DialogueSimulator)
    assert len(runner.interaction_model.agents) == config.num_agents
    assert runner.interaction_model.topic == "A discussion on ice-cream flavors"
    assert isinstance(runner.interaction_model, DialogueSimulator)
    
    for agent in runner.interaction_model.agents:
        assert agent.agent_description, "Agent description not set!"
    assert runner.interaction_model.topic == "A discussion on ice-cream flavors"
    assert runner.interaction_model._step == 0
    assert len(runner.interaction_model.history) == 0


def test_run_rounds(setup_sim_config):
    config = setup_sim_config
    runner = SimulationRunner(config=config, interaction_model=DialogueSimulator)
    runner.run_simulation()

    assert runner.interaction_model._step == config.num_rounds
    assert len(runner.interaction_model.history) == config.num_rounds 
    assert len(runner.interaction_model.agents) == config.num_agents
    
  


@pytest.mark.parametrize("setup_sim_config", params, indirect=True)
def test_run_with_injection(setup_sim_config):
    config = setup_sim_config
    runner = SimulationRunner(config=config, interaction_model=DialogueSimulator)
    runner.interaction_model.inject(
        2, "Hello agents! We are going to discuss our favorite ice-cream flavors."
    )
    runner.run_simulation()

    assert runner.interaction_model._step == config.num_rounds
    assert len(runner.interaction_model.history) == config.num_rounds + 1
    assert len(runner.interaction_model.agents) == config.num_agents


def test_run_with_mediator_injection(setup_sim_config):
    config = setup_sim_config
    runner = SimulationRunner(config=config, interaction_model=DialogueSimulator)
    
    assert runner.interaction_model.mediating_agent.name == "Mediator"   
    assert runner.interaction_model.mediating_agent.system_message is not None
    assert runner.interaction_model.mediating_agent.system_message != ""
    assert runner.interaction_model.mediating_agent.agent_description is not None
    assert (
        runner.interaction_model.mediating_agent.topic_description
        == f"""
        This is a simulated environment where agents communicate with each other. 
        The MediatingAgent initiates and moderates the conversations. 
        It ensures a smooth flow and sets the topic of discussion.
        The topic of discussion is: {config.topic}.
        """
    )
    
    runner.interaction_model.inject(None, "")
    runner.run_simulation()

    assert runner.interaction_model._step == config.num_rounds
    assert len(runner.interaction_model.history) == config.num_rounds + 1
    assert len(runner.interaction_model.agents) == config.num_agents

def test_simulation_runner_with_config():
    """Test the basic configuration setup for SimulationRunner."""
    config = SimulationConfig(
        num_agents=7,
        topic="A discussion on ice-cream flavors",
        num_rounds=10,
        topology=TopologyType.SCALE_FREE,
        model_type=ModelType.GPT3,
        temperature=0.7,
    )

    runner = SimulationRunner(config=config, interaction_model=DialogueSimulator)
    assert runner.config.num_rounds == 10
    assert runner.interaction_model.environment.config.num_agents == 7
    assert len(runner.interaction_model.agents) == 7
    
    for agent in runner.interaction_model.agents:
        assert agent.agent_description, "Agent description not set!"
        assert agent.model is not None
        assert agent.persona is not None
        assert agent.name is not None
        
    assert runner.interaction_model.topic == "A discussion on ice-cream flavors"
    assert runner.interaction_model._step == 0
    assert len(runner.interaction_model.history) == 0
    assert runner.interaction_model.mediating_agent is not None

def test_simulation_runner_with_config_and_local_model():
    """Test SimulationRunner with local Ollama model configuration."""
    config = SimulationConfig(
        num_agents=7,
        topic="A discussion on ice-cream flavors",
        num_rounds=10,
        topology=TopologyType.SCALE_FREE,
        model_type=ModelType.LLAMA2,
        temperature=1.0,
    )

    runner = SimulationRunner(config=config, interaction_model=DialogueSimulator)
    assert runner.interaction_model.environment.config.num_agents == 7
    for agent in runner.interaction_model.agents:
        assert agent.agent_description, "Agent description not set!"
        assert agent.model is not None
        assert agent.model.model == "llama2:13b-chat"
        assert agent.model.temperature == 1.0
    assert runner.interaction_model.topic == "A discussion on ice-cream flavors"
    assert runner.interaction_model._step == 0

@pytest.mark.parametrize("topology", [TopologyType.STAR, TopologyType.SMALL_WORLD, TopologyType.SCALE_FREE])
def test_simulation_runner_with_different_topologies(topology):

    config = SimulationConfig(
        num_agents=5,
        topic="Network topology test",
        num_rounds=3,
        topology=topology,
        model_type=ModelType.GPT3,
        temperature=0.5,
    )
    
    runner = SimulationRunner(config=config, interaction_model=DialogueSimulator)
    assert runner.interaction_model.environment.config.topology == topology
    assert len(runner.interaction_model.agents) == 5
    assert runner.interaction_model.topic == "Network topology test"

@pytest.mark.parametrize("num_agents", [3, 5, 10])
def test_simulation_runner_with_different_agent_counts(num_agents):
    """Test SimulationRunner with different numbers of agents."""
    
    config = SimulationConfig(
            num_agents=num_agents,
            topic="Agent count test",
            num_rounds=2,
            topology=TopologyType.STAR,
            model_type=ModelType.GPT3,
            temperature=0.5,
        )
        
    runner = SimulationRunner(config=config, interaction_model=DialogueSimulator)
    assert len(runner.interaction_model.agents) == num_agents
    assert runner.interaction_model.environment.config.num_agents == num_agents
    
    # Verify all agents have unique names and personas
    agent_names = [agent.name for agent in runner.interaction_model.agents]
    assert len(set(agent_names)) == num_agents  # All names should be unique
