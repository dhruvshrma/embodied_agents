# agents/base_agent.py


class BaseAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id  # Unique identifier for each agent
        self.opinion = 0  # Neutral by default

    def set_opinion(self, opinion):
        self.opinion = opinion

    def get_opinion(self):
        return self.opinion
