from langchain.schema import SystemMessage

from agents.base_agent import BaseAgent
from agents.SimpleAgent import SimpleAgent, MediatingAgent
from agents.GenerativeSocialAgent import GenerativeSocialAgent
import pytest
from utils.log_config import setup_logging, print_to_log

from dotenv import load_dotenv

load_dotenv()
logger = setup_logging()


@pytest.fixture
def language_model():
    from langchain.chat_models import ChatOllama, ChatOpenAI

    # llm = ChatOllama(
    #     model="llama2",
    #     temperature=0.0,
    #     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    # )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    return llm


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


def test_initialize_simple_agent(language_model):
    llm = language_model
    agent = SimpleAgent(name="Mahler", system_message=None, model=llm, agent_id=1)

    assert agent.agent_id == 1
    assert agent.get_opinion() == 0

    assert agent.system_message is None
    assert agent.message_history == ["Here is the conversation so far."]
    assert agent.prefix == "Mahler: "
    agent.system_message = SystemMessage(content="Hello")
    print(agent.send())


def test_initialize_mediating_agent(language_model):
    llm = language_model
    agent = MediatingAgent(name="Mediator", model=llm, agent_id=1)

    assert agent.agent_id == 1
    assert agent.get_opinion() == 0
    assert agent.topic == "A discussion on ice-cream flavors"
    assert agent.message_history == ["Here is the conversation so far."]
    assert agent.prefix == "Mediator: "

    print(agent.system_message)
    agent.set_system_message()
    print(agent.system_message)
    # agent.system_message = SystemMessage(content="Hello")
    # print(agent.send())


def test_store_personal_message_history(language_model):
    agent = SimpleAgent(
        name="Mahler", system_message=None, model=language_model, agent_id=1
    )

    assert agent.agent_id == 1
    assert agent.get_opinion() == 0
    assert agent.personal_message_history == []
    agent.create_agent_description()
    agent.create_system_message(topic="A discussion on ice-cream flavors")

    message = agent.send()

    assert agent.personal_message_history == [message]
