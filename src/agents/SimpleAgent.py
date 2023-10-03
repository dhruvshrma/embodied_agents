from pydantic import Field, root_validator
from agents.base_agent import BaseAgent

from typing import Union
from typing import List, Optional
from langchain.chat_models import ChatOpenAI, ChatOllama

from langchain.schema import (
    HumanMessage,
    SystemMessage,
)


class SimpleAgent(BaseAgent):
    name: str
    system_message: Optional[SystemMessage] = None
    model: Union[ChatOpenAI, ChatOllama]
    prefix: str = Field(default=None)
    message_history: List[str] = Field(
        default_factory=lambda: ["Here is the conversation so far."]
    )

    @root_validator(pre=True)
    def set_prefix(cls, values):
        name = values.get("name")
        values["prefix"] = f"{name}: "
        return values

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        self.message_history.append(f"{name}: {message}")
