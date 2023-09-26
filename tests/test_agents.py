from agents.base_agent import BaseAgent

def test_agent_initialization():
    agent = BaseAgent(agent_id=1)
    assert agent.agent_id == 1
    assert agent.get_opinion() == 0
