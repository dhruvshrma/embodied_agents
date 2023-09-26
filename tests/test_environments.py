from environments.GraphEnvironment import GraphEnvironment
from agents.base_agent import BaseAgent
from hypothesis import given
from hypothesis.strategies import integers


@given(num_agents=integers(min_value=1, max_value=100))
def test_graph_initialization(num_agents):
    env = GraphEnvironment(num_agents=num_agents)
    assert len(env.graph.nodes()) == num_agents
    for i in range(num_agents):
        assert isinstance(env.graph.nodes[i]["agent"], BaseAgent)


def test_graph_agent_opinions():
    env = GraphEnvironment(num_agents=5)
    env.initialize_opinions_randomly()
    for agent in env.agents:
        assert agent.get_opinion() in [-1, 0, 1]
