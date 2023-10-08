from agents.SimpleAgent import SimpleAgent, MediatingAgent
from personas.generate_personas import generate_persona, base_template
from personas.Persona import Persona


class AgentFactory:
    @staticmethod
    def create_simple_agent(persona: Persona) -> SimpleAgent:
        agent = SimpleAgent(name=persona.name, persona=persona)
        agent.create_agent_description()
        return agent

    @staticmethod
    def create_mediating_agent(topic: str) -> MediatingAgent:
        agent = MediatingAgent(name="Mediator", topic=topic)
        agent.set_system_message()
        return agent

    @staticmethod
    def create_random_agent() -> SimpleAgent:
        random_persona = generate_persona(base_template)
        return AgentFactory.create_simple_agent(random_persona)
