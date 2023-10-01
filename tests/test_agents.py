from agents.base_agent import BaseAgent
from agents.GenerativeSocialAgent import GenerativeSocialAgent

from utils.log_config import setup_logging, print_to_log

from dotenv import load_dotenv

load_dotenv()
logger = setup_logging()


def test_agent_initialization():
    agent = BaseAgent(agent_id=1)
    assert agent.agent_id == 1
    assert agent.get_opinion() == 0


def test_initialize_social_agent():
    agent = GenerativeSocialAgent(
        agent_id=1, name="Mahler", age=30, status="single", traits="obnoxious"
    )

    assert agent.agent_id == 1
    assert agent.get_opinion() == 0


def test_initialize_with_language_model(language_model_and_memory):
    agent = GenerativeSocialAgent(
        agent_id=1, name="Mahler", age=30, status="single", traits="obnoxious"
    )

    assert agent.agent_id == 1
    assert agent.get_opinion() == 0
    llm, memory = language_model_and_memory
    agent.llm = llm
    agent.memory = memory

    print_to_log(agent.get_summary())
