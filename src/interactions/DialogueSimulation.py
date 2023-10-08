from typing import List, Callable, Tuple, Optional
from environments.GraphEnvironment import GraphEnvironment
from agents.SimpleAgent import SimpleAgent, MediatingAgent


class DialogueSimulator:
    def __init__(
        self,
        environment: GraphEnvironment,
        selection_function: Callable[[List[SimpleAgent]], int],
        mediating_agent: Optional[MediatingAgent] = None,
        topic: str = "",
    ) -> None:
        self.environment = environment
        self.agents = environment.agents
        self.topic = topic
        self._step = 0
        self.select_next_speaker = selection_function
        self.history = []

        if mediating_agent:
            self.mediating_agent = mediating_agent
            self.mediating_agent.set_topic(topic)
            self.mediating_agent.set_system_message()
        else:
            self.mediating_agent = MediatingAgent(
                name="Mediator", topic="", agent_id=-1
            )
            self.mediating_agent.set_topic(topic)

    def reset(self):
        for agent in self.agents:
            agent.reset()
        self.history.clear()
        self._step = 0

    def inject(self, idx: Optional[int] = None, message: Optional[str] = ""):
        if idx is None:
            name = self.mediating_agent.name
            message = self.mediating_agent.topic_description
        else:
            name = self.agents[idx].name
        for agent in self.agents:
            agent.receive(name, message)
        self._log_interaction(name, message)

    def step(self) -> Tuple[str, str]:
        """Simulate a single step of interaction."""
        # 1. Choose the next speaker
        speaker_idx = self.select_next_speaker(self.agents)
        speaker = self.agents[speaker_idx]

        # 2. Next speaker sends message
        message = speaker.send()

        # 3. Neighbors receive message (assuming neighbors are the ones to receive messages)
        neighbors = self.environment.get_neighbors(speaker)
        for receiver in neighbors:
            receiver.receive(speaker.name, message)

        # 4. Log interaction
        self._log_interaction(speaker.name, message)

        # 5. Increment time step
        self._step += 1

        return speaker.name, message

    def _log_interaction(self, name: str, message: str):
        """Log interactions for future analysis."""
        self.history.append((self._step, name, message))
