class Metrics:
    @staticmethod
    def compute_average_opinion(agents):
        return sum(agent.opinion for agent in agents) / len(agents)

    @staticmethod
    def compute_opinion_distribution(agents):
        positives = sum(1 for agent in agents if agent.opinion == 1)
        neutrals = sum(1 for agent in agents if agent.opinion == 0)
        negatives = sum(1 for agent in agents if agent.opinion == -1)
        return {"positive": positives, "neutral": neutrals, "negative": negatives}
