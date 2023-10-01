from agents.base_agent import BaseAgent
from agents.SimpleAgent import SimpleAgent
from agents.GenerativeSocialAgent import GenerativeSocialAgent


def test_agent_initialization():
    agent = BaseAgent(agent_id=1)
    assert agent.agent_id == 1
    assert agent.get_opinion() == 0


def test_initialize_social_agent():
    agent = GenerativeSocialAgent(
        agent_id=1, name="Mahler", age=30, status="single", traits="obnoxious"
    )

    assert agent.agent_id == 1
    assert agent.get_opinion() == 0
