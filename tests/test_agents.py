from agents.base_agent import BaseAgent
from agents.SimpleAgent import SimpleAgent, MediatingAgent, ModelMissingError
import pytest
from utils.log_config import setup_logging
from configs.configs import LLMConfig, ModelType
from src.llm.openai_client import OpenAIClient
from dotenv import load_dotenv

load_dotenv()
logger = setup_logging()


@pytest.fixture
def language_model():
    llm = OpenAIClient(model="gpt-3.5-turbo", temperature=0.0)
    return llm


@pytest.fixture
def llm_config():
    config = LLMConfig(
        model_type=ModelType.GPT3,
        temperature=0.0,
        presence_penalty=1.0,
        frequency_penalty=1.0,
    )
    return config


def test_agent_initialization():
    agent = BaseAgent(agent_id=1)
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
    agent.system_message = "Hello"
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
    # agent.system_message = "Hello"
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


def test_agent_model_initialization(llm_config):
    agent = SimpleAgent(name="Mahler", system_message=None, model=None, agent_id=1)

    agent.set_model(llm_config)

    assert agent.model is not None
    assert isinstance(agent.model, OpenAIClient)

    assert agent.model.model == "gpt-3.5-turbo"
    assert agent.model.temperature == 0.0


def test_throw_error_without_model_set():
    agent = SimpleAgent(name="Mahler", system_message=None, model=None, agent_id=1)

    with pytest.raises(ModelMissingError):
        agent.create_agent_description()
