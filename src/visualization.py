"""
Módulo de visualização para gerar gráficos prontos para publicação em LaTeX.
"""

import os
import matplotlib.pyplot as plt
import networkx as nx


def _get_positions(G: nx.Graph) -> dict:
    """Usa coordenadas espaciais (x, y) se disponíveis; caso contrário, spring_layout."""
    first_node = next(iter(G.nodes()))
    if "x" in G.nodes[first_node] and "y" in G.nodes[first_node]:
        return {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}
    return nx.spring_layout(G, seed=42)


def plot_structural_vulnerabilities(G: nx.Graph, metrics: dict, save_path: str = None):
    """Destaca pontes (vermelho) e pontos de articulação (laranja) na rede urbana."""
    pos = _get_positions(G)
    art_pts = set(metrics["articulation_points"])
    bridges = {tuple(sorted(e)) for e in metrics["bridges"]}

    fig, ax = plt.subplots(figsize=(10, 7))
    node_colors = ["#e74c3c" if n in art_pts else "#3498db" for n in G.nodes()]
    node_sizes = [120 if n in art_pts else 35 for n in G.nodes()]

    edge_colors = [
        "#c0392b" if tuple(sorted((u, v))) in bridges else "#bdc3c7"
        for u, v in G.edges()
    ]
    edge_widths = [
        2.8 if tuple(sorted((u, v))) in bridges else 0.9
        for u, v in G.edges()
    ]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes)

    ax.set_title("Vulnerabilidades Estruturais: Pontes e Pontos de Articulação", fontsize=13)
    ax.axis("off")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=9, label="Ponto de Articulação"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=7, label="Vértice Comum"),
        plt.Line2D([0], [0], color="#c0392b", lw=2.8, label="Ponte (Aresta Crítica)"),
    ]
    ax.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_betweenness_heatmap(G: nx.Graph, centrality_df, save_path: str = None):
    """Plota mapa de calor da centralidade de intermediação (gargalos de fluxo)."""
    pos = _get_positions(G)
    bc_map = dict(zip(centrality_df["node"], centrality_df["betweenness_centrality"]))
    values = [bc_map.get(n, 0.0) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc", width=0.8)
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=values, cmap="viridis", node_size=65
    )
    plt.colorbar(nodes, ax=ax, label="Centralidade de Intermediação (Betweenness)")
    ax.set_title("Mapa de Gargalos Urbanos (Centralidade de Intermediação)", fontsize=13)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bus_vs_subway(
    G: nx.Graph,
    bus_result: dict,
    subway_result: dict,
    save_path: str = None,
):
    """
    Plota no mesmo mapa:
    - Corredor de Ônibus/BRT (grid 3x3, seguindo as ruas via Dijkstra + Prim)
    - Linha de Metrô Subterrâneo (grid 2x2, em linha reta via Prim)
    """
    pos = _get_positions(G)
    bus_stations = list(dict.fromkeys(bus_result["stations"].values()))
    subway_stations = list(dict.fromkeys(subway_result["stations"].values()))

    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#e0e0e0", width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#bdc3c7", node_size=25)

    # Corredor de Ônibus/BRT (pelas ruas)
    nx.draw_networkx_edges(
        G, pos, ax=ax, edgelist=bus_result["edges"],
        edge_color="#2980b9", width=3.0, label="Ônibus/BRT (Grid 3x3 - Ruas)"
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=bus_stations,
        node_color="#3498db", edgecolors="#1b4f72", node_size=130, label="Paradas Ônibus/BRT (3x3)"
    )

    # Metrô Subterrâneo (linha reta)
    for u, v in subway_result["edges"]:
        x_vals = [pos[u][0], pos[v][0]]
        y_vals = [pos[u][1], pos[v][1]]
        ax.plot(x_vals, y_vals, color="#e74c3c", linewidth=3.2, linestyle="--", zorder=4)

    # Handle manual para a legenda da linha reta do metrô
    ax.plot([], [], color="#e74c3c", linewidth=3.2, linestyle="--", label="Metrô Subterrâneo (Grid 2x2 - Linha Reta)")

    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=subway_stations,
        node_color="#f1c40f", edgecolors="#c0392b", linewidths=2.0,
        node_size=260, label="Estações Metrô (2x2)"
    )

    ax.set_title(
        f"Comparação de Modais: Ônibus/BRT (3x3) vs. Metrô Subterrâneo (2x2) [{bus_result['criterion']}]",
        fontsize=13,
    )
    ax.legend(loc="best", framealpha=0.95)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
