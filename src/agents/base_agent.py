from pydantic import BaseModel, ConfigDict


class BaseAgent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: int
    opinion: int = 0

    def set_opinion(self, opinion: int):
        self.opinion = opinion

    def get_opinion(self) -> int:
        return self.opinion
