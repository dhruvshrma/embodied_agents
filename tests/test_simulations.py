import pytest
from simulation.SimulationRunner import SimulationRunner, random_selector
from environments.GraphEnvironment import GraphEnvironment, GraphEnvironmentConfig
from langchain.chat_models import ChatOpenAI
from agents.SimpleAgent import SimpleAgent
from interactions.DialogueSimulation import DialogueSimulator

from dotenv import load_dotenv

load_dotenv()
params = [("scale-free", 0.0), ("small-world", 0.1), ("small-world", 0.5)]


@pytest.fixture(params=params, scope="function")
def setup_graph_environment(request):
    topology = request.param[0] if request.param else "scale-free"
    temperature = request.param[1] if request.param else 0.0
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
    sim.inject(
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
