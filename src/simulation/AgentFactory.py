from agents.SimpleAgent import SimpleAgent, MediatingAgent
from personas.generate_personas import generate_persona, base_template
from personas.Persona import Persona
from typing import Union
from langchain.chat_models import ChatOpenAI, ChatOllama


class AgentFactory:
    @staticmethod
    def create_simple_agent(persona: Persona, agent_id: int) -> SimpleAgent:
        agent = SimpleAgent(name=persona.name, persona=persona, agent_id=agent_id)
        return agent

    @staticmethod
    def create_mediating_agent(
        topic: str, model: Union[ChatOpenAI, ChatOllama] = None
    ) -> MediatingAgent:
        agent = MediatingAgent(name="Mediator", topic=topic, agent_id=-1, model=model)
        if model is not None:
            agent.set_system_message()
        return agent

    @staticmethod
    def create_random_agent(agent_id: int) -> SimpleAgent:
        random_persona = generate_persona(base_template)
        return AgentFactory.create_simple_agent(random_persona, agent_id=agent_id)
    
    @staticmethod
    def create_testuser_agent(agent_id: int) -> SimpleAgent:
        return SimpleAgent(name="TestUser", agent_id=agent_id)
