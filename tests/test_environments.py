from pydantic import ValidationError
import pytest
from environments.GraphEnvironment import GraphEnvironment, GraphEnvironmentConfig
from hypothesis import given, settings
from hypothesis.strategies import integers, floats


@given(num_agents=integers(min_value=1, max_value=100))
def test_graph_initialization_base(num_agents):
    try:
        config = GraphEnvironmentConfig(num_agents=num_agents, topology="star")
        env = GraphEnvironment(config=config)
        assert len(env.graph.nodes()) == num_agents
        assert env.config.topology == "star"
    except ValidationError:
        pass


def test_graph_initialization_failn1():
    with pytest.raises(ValidationError):
        config = GraphEnvironmentConfig(num_agents=1, topology="star")
        env = GraphEnvironment(config=config)


@given(
    num_agents=integers(min_value=2, max_value=100),
    small_world_k=integers(min_value=1, max_value=100),
    small_world_p=floats(min_value=0.01, max_value=1.0),
)
@settings(deadline=500)
def test_graph_initialization_simple(num_agents, small_world_k, small_world_p):
    try:
        config = GraphEnvironmentConfig(
            num_agents=num_agents,
            topology="small-world",
            small_world_k=small_world_k,
            small_world_p=small_world_p,
        )
        env = GraphEnvironment(config=config)
        assert len(env.graph.nodes()) == num_agents
        assert env.config.topology == "small-world"
        assert env.config.small_world_k == small_world_k
        assert env.config.small_world_p == small_world_p
    except ValidationError:
        pass


def test_edge_case_equaln_k():
    with pytest.raises(ValidationError):
        config = GraphEnvironmentConfig(
            num_agents=3, topology="small-world", small_world_k=3, small_world_p=0.5
        )
        env = GraphEnvironment(config=config)


def test_edge_case_k1_n2():
    with pytest.raises(ValidationError):
        config = GraphEnvironmentConfig(
            num_agents=2, topology="small-world", small_world_k=1, small_world_p=0.5
        )
        env = GraphEnvironment(config=config)
