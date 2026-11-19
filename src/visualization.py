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
    node_sizes = [140 if n in art_pts else 45 for n in G.nodes()]

    edge_colors = [
        "#c0392b" if tuple(sorted((u, v))) in bridges else "#bdc3c7"
        for u, v in G.edges()
    ]
    edge_widths = [
        2.8 if tuple(sorted((u, v))) in bridges else 1.0
        for u, v in G.edges()
    ]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes)

    ax.set_title("Vulnerabilidades Estruturais: Pontes e Pontos de Articulação", fontsize=13)
    ax.axis("off")

    # Legenda
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
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc", width=0.9)
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=values, cmap="viridis", node_size=80
    )
    plt.colorbar(nodes, ax=ax, label="Centralidade de Intermediação (Betweenness)")
    ax.set_title("Mapa de Gargalos Urbanos (Centralidade de Intermediação)", fontsize=13)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_optimized_subway(G: nx.Graph, subway_result: dict, steiner_result: dict = None, save_path: str = None):
    """Plota a rede urbana com a linha de metrô (MST) e, opcionalmente, a Árvore de Steiner."""
    pos = _get_positions(G)
    stations = set(subway_result["stations"].values())

    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#dcdcdc", width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#95a5a6", node_size=35)

    # Linha de Metrô (MST entre estações)
    nx.draw_networkx_edges(
        G, pos, ax=ax, edgelist=subway_result["subway_edges"],
        edge_color="#2980b9", width=3.2, label="Linha de Metrô (Prim/MST)"
    )

    if steiner_result is not None:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=list(steiner_result["steiner_tree"].edges()),
            edge_color="#27ae60", width=2.0, style="dashed", label="Árvore de Steiner (Aprox.)"
        )

    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=list(stations),
        node_color="#f39c12", edgecolors="#2c3e50", node_size=220, label="Estações Selecionadas"
    )

    ax.set_title("Otimização de Infraestrutura: Linha de Metrô vs. Árvore de Steiner", fontsize=13)
    ax.legend(loc="best")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
