from typing import List, Callable, Tuple
from environments.GraphEnvironment import GraphEnvironment
from agents.SimpleAgent import SimpleAgent


class DialogueSimulator:
    def __init__(
        self,
        environment: GraphEnvironment,
        selection_function: Callable[[int, List[SimpleAgent], GraphEnvironment], int],
    ) -> None:
        self.environment = environment
        self.agents = environment.agents
        self._step = 0
        self.select_next_speaker = selection_function
        self.history = []

    def reset(self):
        for agent in self.agents:
            agent.reset()
        self.history.clear()
        self._step = 0

    def inject(self, name: str, message: str):
        """Initiates the conversation with a message from a name."""
        for agent in self.agents:
            agent.receive(name, message)
        self._log_interaction(name, message)

    def step(self) -> Tuple[str, str]:
        """Simulate a single step of interaction."""
        # 1. Choose the next speaker
        speaker_idx = self.select_next_speaker(
            self._step, self.agents, self.environment
        )
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
