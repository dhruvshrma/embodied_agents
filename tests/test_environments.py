from pydantic import ValidationError
import pytest
from environments.GraphEnvironment import GraphEnvironment, GraphEnvironmentConfig
from agents.base_agent import BaseAgent
from agents.SimpleAgent import SimpleAgent
from hypothesis import given
from hypothesis.strategies import integers, floats


@given(num_agents=integers(min_value=1, max_value=100))
def test_graph_initialization_base(num_agents):
    env = GraphEnvironment(num_agents=num_agents)
    assert len(env.graph.nodes()) == num_agents
    for i in range(num_agents):
        assert isinstance(env.graph.nodes[i]["agent"], BaseAgent)


@given(
    num_agents=integers(min_value=2, max_value=100),
    small_world_k=integers(min_value=1, max_value=100),
    small_world_p=floats(min_value=0.01, max_value=1.0),
)
def test_graph_initialization_simple(num_agents, small_world_k, small_world_p):
    try:
        config = GraphEnvironmentConfig(
            num_agents=num_agents,
            topology="small-world",
            small_world_k=small_world_k,
            small_world_p=small_world_p,
        )
        env = GraphEnvironment(agent_class=SimpleAgent, config=config)
        assert len(env.graph.nodes()) == num_agents
        assert env.config.topology == "small-world"
        assert env.config.small_world_k == small_world_k
        assert env.config.small_world_p == small_world_p

        for i in range(num_agents):
            assert isinstance(env.graph.nodes[i]["agent"], SimpleAgent)
    except ValidationError:
        pass


def test_graph_agent_opinions():
    env = GraphEnvironment(num_agents=5)
    env.initialize_opinions_randomly()
    for agent in env.agents:
        assert agent.get_opinion() in [-1, 0, 1]


def test_edge_case_equaln_k():
    with pytest.raises(ValidationError):
        config = GraphEnvironmentConfig(
            num_agents=3, topology="small-world", small_world_k=3, small_world_p=0.5
        )
        env = GraphEnvironment(agent_class=SimpleAgent, config=config)


def test_edge_case_k1_n2():
    with pytest.raises(ValidationError):
        config = GraphEnvironmentConfig(
            num_agents=2, topology="small-world", small_world_k=1, small_world_p=0.5
        )
        env = GraphEnvironment(agent_class=SimpleAgent, config=config)
