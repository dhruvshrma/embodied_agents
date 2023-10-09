from simulation.AgentFactory import AgentFactory
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()


def test_create_mediating_agent():
    agent = AgentFactory.create_mediating_agent(topic="Test topic")
    assert agent.name == "Mediator"
    assert agent.topic == "Test topic"
    assert agent.agent_id == -1

    assert agent.model is None
    assert agent.system_message.content == ""
    assert agent.agent_description is None


def test_create_mediating_agent_model():
    agent = AgentFactory.create_mediating_agent(
        topic="Test topic", model=ChatOpenAI(model="gpt-3.5-turbo", temperature=1.0)
    )
    assert agent.name == "Mediator"
    assert agent.topic == "Test topic"
    assert agent.agent_id == -1

    assert isinstance(agent.model, ChatOpenAI)
    assert agent.system_message.content != ""
    assert agent.agent_description is not None
    assert agent.topic_description is not None


def test_create_simple_agent():
    agent = AgentFactory.create_random_agent(agent_id=1)

    assert agent.name != ""
    assert agent.agent_id == 1
    assert agent.persona is not None
    assert agent.model is None
    assert agent.agent_description is None
    assert agent.system_message.content == ""
