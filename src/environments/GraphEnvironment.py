import networkx as nx
from faker import Faker
from configs.configs import GraphEnvironmentConfig

import plotly.graph_objects as go

fake = Faker()


class GraphEnvironment:
    def __init__(
        self,
        config: GraphEnvironmentConfig,
    ):
        self.config = config
        self.graph = self.create_topology()

    def create_topology(self):
        if self.config.topology == "star":
            return nx.star_graph(self.config.num_agents - 1)
        elif self.config.topology == "small-world":
            return nx.watts_strogatz_graph(
                self.config.num_agents,
                k=self.config.small_world_k,
                p=self.config.small_world_p,
            )
        elif self.config.topology == "scale-free":
            return nx.barabasi_albert_graph(
                self.config.num_agents, m=self.config.scale_free_m
            )

    def get_neighbors(self, agent):
        agent_node = [
            node for node, data in self.graph.nodes(data=True) if data["agent"] == agent
        ][0]
        neighbor_nodes = list(self.graph[agent_node])
        return [self.graph.nodes[node]["agent"] for node in neighbor_nodes]

    def visualize_graph_plotly(self, dimension="2d", k=None):
        if dimension == "2d":
            pos = nx.spring_layout(self.graph, k=k)
        else:
            pos = nx.spring_layout(self.graph, dim=3, k=k)

        edge_traces = []
        for edge in self.graph.edges():
            if dimension == "2d":
                edge_trace = go.Scatter(
                    x=[pos[edge[0]][0], pos[edge[1]][0]],
                    y=[pos[edge[0]][1], pos[edge[1]][1]],
                    mode="lines",
                    line=dict(color="#888", width=0.5),
                    hoverinfo="none",
                )
            else:
                edge_trace = go.Scatter3d(
                    x=[pos[edge[0]][0], pos[edge[1]][0]],
                    y=[pos[edge[0]][1], pos[edge[1]][1]],
                    z=[pos[edge[0]][2], pos[edge[1]][2]],
                    mode="lines",
                    line=dict(color="#888", width=3),
                    hoverinfo="none",
                )
            edge_traces.append(edge_trace)

        if dimension == "2d":
            node_trace = go.Scatter(
                x=[pos[node][0] for node in self.graph.nodes()],
                y=[pos[node][1] for node in self.graph.nodes()],
                mode="markers+text",
                hoverinfo="text",
                marker=dict(size=10, color="skyblue"),
                text=[data["agent"].name for _, data in self.graph.nodes(data=True)],
                textposition="top center",
            )
        else:
            node_trace = go.Scatter3d(
                x=[pos[node][0] for node in self.graph.nodes()],
                y=[pos[node][1] for node in self.graph.nodes()],
                z=[pos[node][2] for node in self.graph.nodes()],
                mode="markers+text",
                hoverinfo="text",
                marker=dict(size=10, color="skyblue"),
                text=[data["agent"].name for _, data in self.graph.nodes(data=True)],
            )

        layout = (
            go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=0, l=0, r=0, t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            if dimension == "2d"
            else {}
        )

        fig = go.Figure(data=edge_traces + [node_trace], layout=layout)
        fig.show()
