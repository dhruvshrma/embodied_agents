from agents.GenerativeSocialAgent import GenerativeSocialAgent
from personas.generate_personas import generate_persona, base_template


class AgentFactory:
    @staticmethod
    def create_agent(
        name: str, age: int, traits: str, status: str
    ) -> GenerativeSocialAgent:
        return GenerativeSocialAgent(name=name, age=age, traits=traits, status=status)

    @staticmethod
    def create_random_agent() -> GenerativeSocialAgent:
        persona = generate_persona(base_template)
        return AgentFactory.create_agent(
            name=persona.name,
            age=persona.age,
            traits=persona.traits,
            status=persona.status,
        )
