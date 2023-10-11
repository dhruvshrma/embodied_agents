from utils.log_config import (
    print_to_log,
)


class DomainEvent:
    pass


class AgentSpoke(DomainEvent):
    def __init__(self, agent_name, message):
        self.agent_name = agent_name
        self.message = message


class EventHandler:
    @staticmethod
    def handle(event: DomainEvent):
        if isinstance(event, AgentSpoke):
            print_to_log("Agent %s said: %s", event.agent_name, event.message)
