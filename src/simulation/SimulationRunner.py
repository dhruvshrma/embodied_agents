import random
from typing import List, Union
from pydantic import BaseModel
from agents.SimpleAgent import SimpleAgent
from interactions.DialogueSimulation import DialogueSimulator
from interactions.VoterModel import VoterModel


def random_selector(agents: List[SimpleAgent]) -> int:
    """A simple selector function that chooses an agent at random."""
    return random.choice(range(len(agents)))


class SimulationRunner(BaseModel):
    num_rounds: int
    interaction_model: Union[DialogueSimulator, VoterModel]

    class Config:
        arbitrary_types_allowed = True

    def run_simulation(self):
        for i in range(self.num_rounds):
            print("----")
            print(f"Round {i}")
            name, message = self.interaction_model.step()
            print(f"{name}: {message}")
            print("----")
