from pydantic import BaseModel


class BaseAgent(BaseModel):
    agent_id: int
    opinion: int = 0

    def set_opinion(self, opinion: int):
        self.opinion = opinion

    def get_opinion(self) -> int:
        return self.opinion
