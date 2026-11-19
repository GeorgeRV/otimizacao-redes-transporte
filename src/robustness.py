"""
Módulo base para análise de robustez e vulnerabilidade da rede urbana.
Simula a remoção de vértices/arestas críticos (ex: pontos de articulação,
pontes ou nós de alta centralidade) e mede o impacto na conectividade e eficiência.
Pronto para ser expandido com cenários "fora da caixa" (enchentes, bloqueios, etc.).
"""

import networkx as nx
import pandas as pd


def simulate_node_removal(G: nx.Graph, nodes_to_remove: list, weight: str = "travel_time") -> dict:
    """
    Remove um conjunto de vértices da rede e compara a conectividade e o tempo
    médio de deslocamento antes e depois da falha.
    """
    G_damaged = G.copy()
    G_damaged.remove_nodes_from(nodes_to_remove)

    orig_eff = nx.global_efficiency(G)
    new_eff = nx.global_efficiency(G_damaged) if G_damaged.number_of_nodes() > 1 else 0.0

    orig_lcc = len(max(nx.connected_components(G), key=len))
    new_lcc = (
        len(max(nx.connected_components(G_damaged), key=len))
        if G_damaged.number_of_nodes() > 0
        else 0
    )

    lcc_graph = G_damaged.subgraph(max(nx.connected_components(G_damaged), key=len))
    orig_avg_path = nx.average_shortest_path_length(G, weight=weight)
    new_avg_path = (
        nx.average_shortest_path_length(lcc_graph, weight=weight)
        if lcc_graph.number_of_nodes() > 1
        else float("inf")
    )

    return {
        "removed_nodes": nodes_to_remove,
        "original_components": nx.number_connected_components(G),
        "new_components": nx.number_connected_components(G_damaged),
        "largest_cc_ratio": round(new_lcc / orig_lcc, 4),
        "efficiency_drop_pct": round((1 - new_eff / orig_eff) * 100, 2) if orig_eff > 0 else 0.0,
        "original_avg_travel_time": round(orig_avg_path, 2),
        "new_avg_travel_time_lcc": round(new_avg_path, 2),
    }


def targeted_attack_curve(G: nx.Graph, k_steps: int = 5, weight: str = "travel_time") -> pd.DataFrame:
    """
    Avalia a degradação progressiva da rede ao remover sequencialmente os `k_steps`
    vértices com maior centralidade de intermediação (gargalos da cidade).
    """
    bc = nx.betweenness_centrality(G, weight=weight)
    ranked_nodes = [n for n, _ in sorted(bc.items(), key=lambda x: x[1], reverse=True)]

    records = []
    for k in range(1, min(k_steps, len(ranked_nodes)) + 1):
        res = simulate_node_removal(G, ranked_nodes[:k], weight=weight)
        records.append(
            {
                "nodes_removed": k,
                "components": res["new_components"],
                "largest_cc_ratio": res["largest_cc_ratio"],
                "efficiency_drop_pct": res["efficiency_drop_pct"],
            }
        )
    return pd.DataFrame(records)
