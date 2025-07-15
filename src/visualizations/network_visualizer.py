import networkx as nx
import plotly.graph_objects as go
import os


def visualize_graph(G, title, output_dir="output"):
    pos = nx.spring_layout(G, dim=2)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            size=12,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
        ),
    )

    # Color nodes by degree
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f"# of connections: {len(adjacencies[1])}")

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            width=800,
            height=800,
            plot_bgcolor="#f0f0f0",
            annotations=[
                dict(showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig.write_html(os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.html"))


if __name__ == "__main__":
    n_nodes = 100

    # Star Graph
    star_graph = nx.star_graph(n_nodes - 1)
    visualize_graph(star_graph, title="Star Network")

    # Small-World Graph
    small_world_graph = nx.watts_strogatz_graph(n=n_nodes, k=4, p=0.1)
    visualize_graph(small_world_graph, title="Small-World Network")

    # Scale-Free Graph
    scale_free_graph = nx.barabasi_albert_graph(n=n_nodes, m=2)
    visualize_graph(scale_free_graph, title="Scale-Free Network")
