# agents/special_agent.py

from agents.base_agent import BaseAgent


class SimpleAgent(BaseAgent):
    def __init__(self, agent_id, special_attribute):
        self.agent_id = agent_id
        self.special_attribute = special_attribute

    def set_opinion(self, opinion):
        self.opinion = opinion

    def get_opinion(self):
        return self.opinion
