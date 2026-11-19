"""
Módulo de análise estrutural de redes urbanas.
Calcula conectividade, componentes conexas, pontes, pontos de articulação
e medidas de centralidade (Grau, Proximidade e Intermediação).
"""

import networkx as nx
import pandas as pd


def compute_structural_metrics(G: nx.Graph, weight: str = "length") -> dict:
    """
    Calcula as principais propriedades estruturais da rede urbana.
    """
    is_conn = nx.is_connected(G)
    num_cc = nx.number_connected_components(G)
    bridges = list(nx.bridges(G))
    articulation_pts = list(nx.articulation_points(G))

    # Eficiência global (média do inverso das distâncias mínimas entre pares)
    global_eff = nx.global_efficiency(G)

    # Diâmetro e caminho médio na maior componente conexa
    largest_cc = G.subgraph(max(nx.connected_components(G), key=len))
    avg_shortest_path = nx.average_shortest_path_length(largest_cc, weight=weight)
    diameter = nx.diameter(largest_cc)

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": round(nx.density(G), 5),
        "is_connected": is_conn,
        "num_connected_components": num_cc,
        "num_bridges": len(bridges),
        "bridges": bridges,
        "num_articulation_points": len(articulation_pts),
        "articulation_points": articulation_pts,
        "global_efficiency": round(global_eff, 5),
        "avg_shortest_path_length": round(avg_shortest_path, 2),
        "diameter_hops": diameter,
    }


def compute_centralities(G: nx.Graph, weight: str = "travel_time") -> pd.DataFrame:
    """
    Calcula centralidades de Grau, Proximidade (Closeness) e Intermediação (Betweenness)
    para todos os vértices do grafo.
    """
    deg = nx.degree_centrality(G)
    closeness = nx.closeness_centrality(G, distance=weight)
    betweenness = nx.betweenness_centrality(G, weight=weight, normalized=True)

    rows = []
    for n in G.nodes():
        rows.append(
            {
                "node": n,
                "region": G.nodes[n].get("region", "N/A"),
                "type": G.nodes[n].get("type", "N/A"),
                "degree": G.degree(n),
                "degree_centrality": round(deg[n], 5),
                "closeness_centrality": round(closeness[n], 5),
                "betweenness_centrality": round(betweenness[n], 5),
            }
        )

    df = pd.DataFrame(rows).sort_values(by="betweenness_centrality", ascending=False)
    return df.reset_index(drop=True)
