"""
Diagnóstico estrutural breve da malha viária urbana e cálculo de centralidade.
Fornece métricas básicas de conectividade/redundância (pontes e pontos de articulação)
e a centralidade de intermediação (Betweenness) para visualização dos gargalos da cidade.
"""

import networkx as nx
import pandas as pd


def compute_network_summary(G: nx.Graph, weight: str = "length") -> dict:
    """Retorna um resumo objetivo das propriedades estruturais da malha viária."""
    bridges = list(nx.bridges(G))
    art_pts = list(nx.articulation_points(G))
    largest_cc = G.subgraph(max(nx.connected_components(G), key=len))

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": round(nx.density(G), 5),
        "is_connected": nx.is_connected(G),
        "num_bridges": len(bridges),
        "num_articulation_points": len(art_pts),
        "avg_shortest_path_m": round(nx.average_shortest_path_length(largest_cc, weight=weight), 2),
        "diameter_hops": nx.diameter(largest_cc),
    }


def compute_centralities(G: nx.Graph, weight: str = "travel_time") -> pd.DataFrame:
    """Calcula a centralidade de intermediação (Betweenness) para visualização dos gargalos."""
    betweenness = nx.betweenness_centrality(G, weight=weight, normalized=True)
    rows = [
        {"node": n, "betweenness_centrality": round(betweenness[n], 5)}
        for n in G.nodes()
    ]
    return pd.DataFrame(rows).sort_values(by="betweenness_centrality", ascending=False).reset_index(drop=True)
