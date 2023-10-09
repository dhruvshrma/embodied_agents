from typing import List, Callable, Tuple, Optional
from environments.GraphEnvironment import GraphEnvironment
from agents.SimpleAgent import SimpleAgent, MediatingAgent


class DialogueSimulator:
    def __init__(
        self,
        environment: GraphEnvironment,
        selection_function: Callable[[List[SimpleAgent]], int],
        agents: List[SimpleAgent],
        mediating_agent: Optional[MediatingAgent] = None,
        topic: str = "",
    ) -> None:
        self.environment = environment
        self.agents = agents
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
        combined_list = self.agents + [self.mediating_agent]
        speaker_idx = self.select_next_speaker(combined_list)
        # speaker = self.agents[speaker_idx]
        speaker = combined_list[speaker_idx]
        message = speaker.send()
        if isinstance(speaker, MediatingAgent):
            # If the speaker is the mediating agent, all agents receive the message
            receivers = self.agents
        else:
            # If a regular agent is speaking, only their neighbors receive the message
            receivers = self.environment.get_neighbors(speaker)
            # Ensure the MediatingAgent also receives the message
            if self.mediating_agent not in receivers:
                receivers.append(self.mediating_agent)

        for receiver in receivers:
            receiver.receive(speaker.name, message)

        # 4. Log interaction
        self._log_interaction(speaker.name, message)

        # 5. Increment time step
        self._step += 1

        return speaker.name, message

    def _log_interaction(self, name: str, message: str):
        """Log interactions for future analysis."""
        self.history.append((self._step, name, message))
