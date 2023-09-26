from base_interaction import BaseInteraction


class MajorityRule(BaseInteraction):
    def interact(self, agent, neighbors):
        positive_count = sum(1 for neighbor in neighbors if neighbor.opinion == 1)
        negative_count = sum(1 for neighbor in neighbors if neighbor.opinion == -1)

        if positive_count > negative_count:
            agent.set_opinion(1)
        elif negative_count > positive_count:
            agent.set_opinion(-1)
        else:
            agent.set_opinion(0)
