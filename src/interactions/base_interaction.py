from abc import ABC, abstractmethod


class BaseInteraction(ABC):
    @abstractmethod
    def interact(self, agent, neighbors):
        pass
