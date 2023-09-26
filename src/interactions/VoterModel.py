from interactions.base_interaction import BaseInteraction
import random


class VoterModel(BaseInteraction):
    def interact(self, agent, neighbors):
        chosen_neighbor = random.choice(neighbors)
        agent.set_opinion(chosen_neighbor.get_opinion())
