# agents/special_agent.py

from agents.base_agent import BaseAgent


class SimpleAgent(BaseAgent):
    def __init__(self, agent_id, special_attribute):
        super().__init__(agent_id)
        self.special_attribute = None
