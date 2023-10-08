from pydantic import Field, root_validator
from agents.base_agent import BaseAgent
from personas.Persona import Persona
from personas.generate_personas import generate_persona, base_template

from typing import Union
from typing import List, Optional
from langchain.chat_models import ChatOpenAI, ChatOllama

from langchain.schema import (
    HumanMessage,
    SystemMessage,
)


class SimpleAgent(BaseAgent):
    name: str
    system_message: Optional[SystemMessage] = SystemMessage(content="")
    model: Optional[Union[ChatOpenAI, ChatOllama]] = None
    prefix: str = Field(default=None)
    message_history: List[str] = Field(
        default_factory=lambda: ["Here is the conversation so far."]
    )
    persona: Optional[Persona] = None
    # base_descriptor_system_message: SystemMessage = SystemMessage(content="")
    subject_description: str = (
        "You are an agent participating in a social dynamics simulation."
    )
    agent_description: Optional[str] = None

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

    def create_agent_description(self) -> None:
        if self.persona is None:
            self.persona = generate_persona(base_template)
            self.persona.name = self.name
        agent_specifier_prompt = [
            # self.base_descriptor_system_message,
            HumanMessage(
                content=f"""{self.subject_description}
                Please provide a description for {self.persona.name}, a {self.persona.age}-year-old {self.persona.status} with {self.persona.traits} traits.
                Do not add anything else."""
            ),
        ]
        agent_description = self.model(agent_specifier_prompt).content
        self.agent_description = agent_description

    def generate_character_system_message(
        self, agent_description: str, topic: str
    ) -> SystemMessage:
        return SystemMessage(
            content=f"""
            {self.subject_description}
            Your name is {self.name}. Your description is as follows: {agent_description}.
            You will be discussing the topic of {topic} and you will be able to present your views to the other agents.
            You will also be able to respond to the views of the other agents.
            Speak in the first person from the perspective of {self.name}.
            Do not change roles!
            Do not speak from the perspective of anyone else.
            Stop speaking the moment you finish speaking from your perspective.
            Do not add anything else.
            """
        )

    def create_system_message(self, topic: str = "A discussion on ice-cream flavors"):
        self.system_message = self.generate_character_system_message(
            self.agent_description, topic
        )


class MediatingAgent(SimpleAgent):
    topic: Optional[str] = "A discussion on ice-cream flavors"
    topic_description: Optional[str] = ""

    def set_topic(self, topic: str):
        self.topic = topic

    def set_system_message(self):
        simulation_description = f"""
        This is a simulated environment where agents communicate with each other. 
        The MediatingAgent initiates and moderates the conversations. 
        It ensures a smooth flow and sets the topic of discussion.
        The topic of discussion is: {self.topic}.
        """

        self.topic_description = simulation_description
        mediating_agent_specifier_prompt = [
            SystemMessage(content=simulation_description),
            HumanMessage(
                content=f"""Please describe the MediatingAgent's role and characteristics in this simulation in 100 words or less."""
            ),
        ]

        mediating_agent_description = self.model(
            mediating_agent_specifier_prompt
        ).content

        self.agent_description = mediating_agent_description

        self.system_message = SystemMessage(
            content=(
                f"""{simulation_description}
                You are the MediatingAgent.
                Your role and characteristics are as follows: {mediating_agent_description}.
                You initiate and moderate the conversations among the agents.
                Speak in the first person from the perspective of the MediatingAgent.
                Do not change roles!
                Do not speak from the perspective of anyone else.
                Remember you are the MediatingAgent.
                Stop speaking the moment you finish speaking from your perspective.
                Keep your responses concise.
                Do not add anything else.
                """
            )
        )
