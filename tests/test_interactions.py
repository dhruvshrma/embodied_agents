from agents.base_agent import BaseAgent
from interactions.VoterModel import VoterModel


def test_voter_model_interaction():
    agent1 = BaseAgent(agent_id=1)
    agent2 = BaseAgent(agent_id=2)
    agent1.set_opinion(1)
    agent2.set_opinion(-1)

    vm = VoterModel()
    vm.interact(agent1, [agent2])
    assert agent1.get_opinion() == -1
