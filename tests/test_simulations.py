import pytest
from simulation.SimulationRunner import SimulationRunner, random_selector
from environments.GraphEnvironment import GraphEnvironment, GraphEnvironmentConfig
from langchain.chat_models import ChatOpenAI
from agents.SimpleAgent import SimpleAgent, MediatingAgent
from interactions.DialogueSimulation import DialogueSimulator
from rich import print
from dotenv import load_dotenv

load_dotenv()
params = [("scale-free", 1.0), ("small-world", 0.5), ("small-world", 1.0)]


@pytest.fixture(params=params, scope="function")
def setup_graph_environment(request):
    topology = request.param[0] if request.param else "scale-free"
    temperature = request.param[1] if request.param else 1.0
    print(f"Running test with topology={topology} and temperature={temperature}")
    config = GraphEnvironmentConfig(
        num_agents=7, topology=topology, small_world_k=2, small_world_p=0.3
    )
    env = GraphEnvironment(agent_class=SimpleAgent, config=config)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
    for agent in env.agents:
        agent.model = llm
    sim = DialogueSimulator(environment=env, selection_function=random_selector)

    return env, sim


def test_initialization(setup_graph_environment):
    env, sim = setup_graph_environment
    runner = SimulationRunner(num_rounds=10, interaction_model=sim)
    assert runner.num_rounds == 10
    assert isinstance(runner.interaction_model, DialogueSimulator)


def test_run_rounds(setup_graph_environment):
    env, sim = setup_graph_environment
    runner = SimulationRunner(num_rounds=10, interaction_model=sim)
    runner.run_simulation()

    assert runner.interaction_model._step == 10
    assert len(runner.interaction_model.history) == 10
    assert len(runner.interaction_model.agents) == 3


@pytest.mark.parametrize("setup_graph_environment", params, indirect=True)
def test_run_with_injection(setup_graph_environment):
    env, sim = setup_graph_environment
    runner = SimulationRunner(num_rounds=10, interaction_model=sim)
    runner.interaction_model.inject(
        2, "Hello agents! We are going to discuss our favorite ice-cream flavors."
    )
    runner.run_simulation()

    assert runner.interaction_model._step == runner.num_rounds
    assert len(runner.interaction_model.history) == runner.num_rounds + 1
    assert len(runner.interaction_model.agents) == env.config.num_agents

    # random_agent = random_selector(runner.interaction_model.agents)
    # print(
    #     f"History for {runner.interaction_model.agents[random_agent]}: {runner.interaction_model.agents[random_agent].message_history}"
    # )


def test_run_with_mediator_injection(setup_graph_environment):
    env, sim = setup_graph_environment
    sim.mediating_agent = MediatingAgent(name="Mediator", agent_id=-1)
    sim.mediating_agent.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=1.0)
    sim.mediating_agent.set_system_message()

    assert sim.mediating_agent.system_message is not None
    assert sim.mediating_agent.system_message.content is not None
    assert sim.mediating_agent.agent_description is not None
    assert (
        sim.mediating_agent.topic_description
        == f"""
        This is a simulated environment where agents communicate with each other. 
        The MediatingAgent initiates and moderates the conversations. 
        It ensures a smooth flow and sets the topic of discussion.
        The topic of discussion is: {sim.mediating_agent.topic}.
        """
    )

    runner = SimulationRunner(num_rounds=10, interaction_model=sim)
    runner.interaction_model.inject(None, "")
    runner.run_simulation()

    # print(
    assert runner.interaction_model._step == runner.num_rounds
    assert len(runner.interaction_model.history) == runner.num_rounds + 1
    assert len(runner.interaction_model.agents) == env.config.num_agents

    # random_agent = random_selector(runner.interaction_model.agents)
    # print(
    #     f"History for {runner.interaction_model.agents[random_agent]}: {runner.interaction_model.agents[random_agent].message_history}"
    #
