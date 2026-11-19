"""
Módulo de visualização para o relatório em LaTeX (300 DPI).
Gera:
1. Mapa da infraestrutura planejada: Ônibus/BRT (3x3 - Ruas) vs Metrô (2x2 - Linha Reta).
2. Comparativo visual de rotas multimodais para diferentes orçamentos (ex: R$ 0, R$ 10, R$ 20, R$ 40).
"""

import os
import matplotlib.pyplot as plt
import networkx as nx


def _get_positions(G: nx.Graph) -> dict:
    first_node = next(iter(G.nodes()))
    if "x" in G.nodes[first_node] and "y" in G.nodes[first_node]:
        return {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}
    return nx.spring_layout(G, seed=42)


def plot_bus_vs_subway(G: nx.Graph, bus_result: dict, subway_result: dict, save_path: str = None):
    """Plota Ônibus/BRT (grid 3x3 nas ruas) e Metrô Subterrâneo (grid 2x2 em linha reta)."""
    pos = _get_positions(G)
    bus_stations = list(dict.fromkeys(bus_result["stations"].values()))
    subway_stations = list(dict.fromkeys(subway_result["stations"].values()))

    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#e0e0e0", width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#bdc3c7", node_size=22)

    nx.draw_networkx_edges(
        G, pos, ax=ax, edgelist=bus_result["edges"],
        edge_color="#2980b9", width=3.0, label="Ônibus/BRT (Grid 3x3 - Ruas)"
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=bus_stations,
        node_color="#3498db", edgecolors="#1b4f72", node_size=120, label="Paradas Ônibus/BRT (3x3)"
    )

    for u, v in subway_result["edges"]:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#e74c3c", linewidth=3.2, linestyle="--", zorder=4)
    ax.plot([], [], color="#e74c3c", linewidth=3.2, linestyle="--", label="Metrô Subterrâneo (Grid 2x2 - Linha Reta)")

    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=subway_stations,
        node_color="#f1c40f", edgecolors="#c0392b", linewidths=2.0, node_size=250, label="Estações Metrô (2x2)"
    )

    ax.set_title("Infraestrutura Planejada: Ônibus/BRT (3x3) vs. Metrô Subterrâneo (2x2)", fontsize=13)
    ax.legend(loc="best", framealpha=0.95)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_budget_routes_comparison(
    G: nx.Graph,
    routes_list: list,
    source_node,
    target_node,
    save_path: str = None,
):
    """Plota lado a lado as rotas ótimas encontradas para cada faixa de orçamento."""
    pos = _get_positions(G)
    n_plots = len(routes_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.2 * n_plots, 5.5))
    if n_plots == 1:
        axes = [axes]

    mode_styles = {
        "walk": {"color": "#7f8c8d", "style": ":", "width": 2.5, "label": "A pé"},
        "bus": {"color": "#2980b9", "style": "-", "width": 3.2, "label": "Ônibus/BRT"},
        "subway": {"color": "#e74c3c", "style": "--", "width": 3.5, "label": "Metrô"},
        "car": {"color": "#8e44ad", "style": "-.", "width": 3.0, "label": "Carro/App"},
    }

    for ax, res in zip(axes, routes_list):
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#ecf0f1", width=0.7)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#d5dbdb", node_size=12)

        for u, v, mode in res["segments"]:
            st = mode_styles[mode]
            ax.plot(
                [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=st["color"], linestyle=st["style"], linewidth=st["width"], zorder=4
            )

        ax.scatter(*pos[source_node], c="#27ae60", s=160, edgecolors="black", zorder=5, label="Origem (A)")
        ax.scatter(*pos[target_node], c="#e67e22", s=160, edgecolors="black", zorder=5, label="Destino (B)")

        title = (
            f"Orçamento <= R$ {res['budget_limit']:.0f}\n"
            f"Tempo: {res['total_time_min']} min | Gasto: R$ {res['total_fare_brl']:.2f}\n"
            f"Modais: {res['modes_sequence']}"
        )
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Legenda global
    handles = [
        plt.Line2D([0], [0], color=st["color"], linestyle=st["style"], lw=3, label=st["label"])
        for st in mode_styles.values()
    ]
    handles.extend([
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#27ae60", markeredgecolor="black", markersize=9, label="Origem (A)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22", markeredgecolor="black", markersize=9, label="Destino (B)"),
    ])
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=True, fontsize=10)
    plt.suptitle("Guia de Rotas Multimodal com Restrição de Orçamento (Dijkstra)", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])

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
