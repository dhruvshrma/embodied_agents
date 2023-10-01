from agents.base_agent import BaseAgent
from agents.GenerativeAgent import GenerativeAgent


class GenerativeSocialAgent(GenerativeAgent, BaseAgent):
    agent_id: int
    opinion: int = 0
