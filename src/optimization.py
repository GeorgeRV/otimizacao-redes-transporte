"""
Módulo de otimização de infraestrutura e rotas em redes de transporte urbano.
Inclui:
- Migração do algoritmo original em C++ (melhor estação por região via centro do subgrafo + linha de metrô)
- Árvore Geradora Mínima (MST - Prim e Kruskal)
- Árvore de Steiner aproximada (conectando pontos de interesse com menor custo)
- Caminhos mínimos multi-critério (distância, tempo e custo)
"""

import time
import networkx as nx
from networkx.algorithms.approximation import steiner_tree


def find_best_station_in_region(G: nx.Graph, region_nodes: list, weight: str = "length"):
    """
    Replica a lógica do método `findBestStation` do projeto original em C++.
    Encontra o vértice dentro da região cuja distância máxima para qualquer outro
    vértice da mesma região é mínima (centro do subgrafo / menor excentricidade).
    """
    subG = G.subgraph(region_nodes)
    if not nx.is_connected(subG):
        largest = max(nx.connected_components(subG), key=len)
        subG = subG.subgraph(largest)

    best_station = None
    min_eccentricity = float("inf")

    for v in subG.nodes():
        lengths = nx.single_source_dijkstra_path_length(subG, v, weight=weight)
        ecc = max(lengths.values())
        if ecc < min_eccentricity:
            min_eccentricity = ecc
            best_station = v

    return best_station, min_eccentricity


def design_subway_line_mst(
    G: nx.Graph,
    regions: dict,
    station_weight: str = "length",
    infra_weight: str = "cost",
) -> dict:
    """
    Replica e aprimora `defineSubwayLine` do projeto original em C++.
    1. Seleciona a melhor estação em cada região.
    2. Constrói o grafo completo métrico entre as estações usando Dijkstra (menor custo).
    3. Aplica o algoritmo de Prim para obter a Árvore Geradora Mínima das estações.
    4. Reconstrói as arestas reais da cidade que compõem a linha de metrô.
    """
    t0 = time.perf_counter()

    stations = {}
    for reg_id, nodes in regions.items():
        if nodes:
            st, ecc = find_best_station_in_region(G, nodes, weight=station_weight)
            stations[reg_id] = st

    station_list = list(stations.values())
    metric_graph = nx.Graph()
    paths_cache = {}

    for i in range(len(station_list)):
        u = station_list[i]
        dists, paths = nx.single_source_dijkstra(G, u, weight=infra_weight)
        for j in range(i + 1, len(station_list)):
            v = station_list[j]
            if v in dists:
                metric_graph.add_edge(u, v, weight=dists[v])
                paths_cache[(u, v)] = paths[v]
                paths_cache[(v, u)] = list(reversed(paths[v]))

    mst_stations = nx.minimum_spanning_tree(metric_graph, weight="weight", algorithm="prim")

    subway_edges = set()
    for u, v in mst_stations.edges():
        path = paths_cache[(u, v)]
        for k in range(len(path) - 1):
            edge = tuple(sorted((path[k], path[k + 1])))
            subway_edges.add(edge)

    subway_subgraph = G.edge_subgraph(subway_edges).copy()
    total_cost = sum(G[u][v][infra_weight] for u, v in subway_edges)
    total_length = sum(G[u][v]["length"] for u, v in subway_edges)
    elapsed = time.perf_counter() - t0

    return {
        "stations": stations,
        "subway_subgraph": subway_subgraph,
        "subway_edges": list(subway_edges),
        "total_cost": round(total_cost, 2),
        "total_length": round(total_length, 2),
        "execution_time_s": round(elapsed, 6),
    }


def compute_city_mst(G: nx.Graph, weight: str = "cost", algorithm: str = "kruskal") -> dict:
    """Calcula a Árvore Geradora Mínima (MST) de toda a rede urbana."""
    t0 = time.perf_counter()
    mst = nx.minimum_spanning_tree(G, weight=weight, algorithm=algorithm)
    elapsed = time.perf_counter() - t0
    total_weight = sum(d[weight] for _, _, d in mst.edges(data=True))
    return {
        "mst": mst,
        "algorithm": algorithm,
        "total_weight": round(total_weight, 2),
        "num_edges": mst.number_of_edges(),
        "execution_time_s": round(elapsed, 6),
    }


def compute_steiner_network(G: nx.Graph, terminal_nodes: list, weight: str = "cost") -> dict:
    """
    Calcula uma aproximação da Árvore de Steiner (algoritmo de Mehlhorn/Kou)
    para conectar um conjunto de pontos de interesse (terminais) com custo reduzido.
    """
    t0 = time.perf_counter()
    st_tree = steiner_tree(G, terminal_nodes, weight=weight)
    elapsed = time.perf_counter() - t0
    total_cost = sum(d[weight] for _, _, d in st_tree.edges(data=True))
    total_length = sum(d.get("length", 0.0) for _, _, d in st_tree.edges(data=True))
    return {
        "steiner_tree": st_tree,
        "terminals": terminal_nodes,
        "total_cost": round(total_cost, 2),
        "total_length": round(total_length, 2),
        "num_edges": st_tree.number_of_edges(),
        "execution_time_s": round(elapsed, 6),
    }


def compare_multicriteria_routes(G: nx.Graph, source, target) -> dict:
    """
    Compara rotas ótimas (Dijkstra) sob diferentes critérios:
    menor distância (length), menor tempo (travel_time) e menor custo (cost).
    """
    results = {}
    for criterion in ["length", "travel_time", "cost"]:
        path = nx.shortest_path(G, source=source, target=target, weight=criterion)
        total_len = sum(G[path[i]][path[i + 1]]["length"] for i in range(len(path) - 1))
        total_time = sum(G[path[i]][path[i + 1]]["travel_time"] for i in range(len(path) - 1))
        total_cost = sum(G[path[i]][path[i + 1]]["cost"] for i in range(len(path) - 1))
        results[criterion] = {
            "path": path,
            "hops": len(path) - 1,
            "length_m": round(total_len, 2),
            "travel_time_s": round(total_time, 2),
            "cost": round(total_cost, 2),
        }
    return results
