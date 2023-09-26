from environments.GraphEnvironment import GraphEnvironment


class SimulationRunner:
    def __init__(self, num_agents, topology, interaction_model, num_rounds):
        self.environment = GraphEnvironment(topology=topology, num_agents=num_agents)
        self.interaction_model = interaction_model
        self.num_rounds = num_rounds

    def run_simulation(self):
        for _ in range(self.num_rounds):
            for agent in self.environment.agents:
                neighbors = self.environment.get_neighbors(agent)
                self.interaction_model.interact(agent, neighbors)
