from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class GenerativeAgent(BaseModel):
    name: str
    age: int
    traits: str
    status: str

    summary: str = ""  #: :meta private:
    """Stateful self-summary generated via reflection on the character's memory."""
    summary_refresh_seconds: int = 3600  #: :meta private:
    """How frequently to re-generate the summary."""
    last_refreshed: datetime = Field(default_factory=datetime.now)  # : :meta private:
    """The last time the character's summary was regenerated."""
    daily_summaries: List[str] = Field(default_factory=list)  # : :meta private:
    """Summary of the events in the plan that the agent took."""
