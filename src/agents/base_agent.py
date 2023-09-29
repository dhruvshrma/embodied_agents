from pydantic import BaseModel


class BaseAgent(BaseModel):
    agent_id: int
    opinion: int = 0

    def set_opinion(self, opinion):
        self.opinion = opinion

    def get_opinion(self):
        return self.opinion
