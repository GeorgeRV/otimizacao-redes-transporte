"""
Módulo de otimização de infraestrutura e rotas em redes de transporte urbano.
Compara:
- Seleção de estações por 3 critérios de centralidade:
  * 'eccentricity' (Equidade / Centro do subgrafo - menor distância máxima)
  * 'closeness' (Eficiência média - centralidade de proximidade)
  * 'betweenness' (Fluxo - centralidade de intermediação)
- Modo Ônibus/BRT (segue as ruas existentes via Dijkstra + Prim/MST)
- Modo Metrô Subterrâneo (ignora as ruas e liga estações em linha reta via distância direta + Prim/MST)
- Árvore de Steiner aproximada e caminhos mínimos multi-critério
"""

import math
import time
import networkx as nx
from networkx.algorithms.approximation import steiner_tree


def _direct_distance_meters(G: nx.Graph, u, v) -> float:
    """
    Calcula a distância em linha reta (metros) entre dois nós u e v.
    Usa fórmula de Haversine se o grafo for geográfico (lat/lon do OSM);
    caso contrário, usa distância euclidiana nas coordenadas (x, y).
    """
    x1, y1 = float(G.nodes[u].get("x", 0.0)), float(G.nodes[u].get("y", 0.0))
    x2, y2 = float(G.nodes[v].get("x", 0.0)), float(G.nodes[v].get("y", 0.0))

    if G.graph.get("is_geographic", False):
        R = 6371000.0  # Raio médio da Terra em metros
        phi1, phi2 = math.radians(y1), math.radians(y2)
        dphi = math.radians(y2 - y1)
        dlambda = math.radians(x2 - x1)
        a = (
            math.sin(dphi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        )
        return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return math.hypot(x2 - x1, y2 - y1)


def find_best_station_in_region(
    G: nx.Graph,
    region_nodes: list,
    criterion: str = "eccentricity",
    weight: str = "length",
):
    """
    Seleciona a melhor estação dentro de uma região segundo o critério escolhido:
    - 'eccentricity': menor distância máxima até qualquer outro nó da região (equidade)
    - 'closeness': maior centralidade de proximidade na região (eficiência média)
    - 'betweenness': maior centralidade de intermediação na região (gargalo de fluxo)
    """
    subG = G.subgraph(region_nodes)
    if subG.number_of_nodes() == 0:
        return None, 0.0
    if not nx.is_connected(subG):
        largest = max(nx.connected_components(subG), key=len)
        subG = subG.subgraph(largest)

    if criterion == "eccentricity":
        best_station = None
        min_ecc = float("inf")
        for v in subG.nodes():
            lengths = nx.single_source_dijkstra_path_length(subG, v, weight=weight)
            ecc = max(lengths.values()) if lengths else 0.0
            if ecc < min_ecc:
                min_ecc = ecc
                best_station = v
        return best_station, round(min_ecc, 2)

    if criterion == "closeness":
        cc = nx.closeness_centrality(subG, distance=weight)
        best_station = max(cc, key=cc.get)
        return best_station, round(cc[best_station], 5)

    if criterion == "betweenness":
        bc = nx.betweenness_centrality(subG, weight=weight, normalized=True)
        best_station = max(bc, key=bc.get)
        return best_station, round(bc[best_station], 5)

    raise ValueError(f"Critério inválido: {criterion}. Use 'eccentricity', 'closeness' ou 'betweenness'.")


def design_bus_network_mst(
    G: nx.Graph,
    regions: dict,
    criterion: str = "eccentricity",
    station_weight: str = "length",
    infra_weight: str = "cost",
) -> dict:
    """
    Modo Ônibus/BRT (superfície):
    1. Seleciona 1 parada/estação por região (ex: grid 3x3).
    2. Roda Dijkstra pelas ruas da cidade para achar o menor custo entre pares de paradas.
    3. Aplica Prim (MST) no grafo métrico das paradas e reconstrói as arestas reais das ruas.
    """
    t0 = time.perf_counter()

    stations = {}
    for reg_id, nodes in sorted(regions.items()):
        if nodes:
            st, _ = find_best_station_in_region(G, nodes, criterion=criterion, weight=station_weight)
            if st is not None:
                stations[reg_id] = st

    station_list = list(dict.fromkeys(stations.values()))
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

    bus_edges = set()
    for u, v in mst_stations.edges():
        path = paths_cache[(u, v)]
        for k in range(len(path) - 1):
            bus_edges.add(tuple(sorted((path[k], path[k + 1]))))

    bus_subgraph = G.edge_subgraph(bus_edges).copy()
    total_cost = sum(G[u][v][infra_weight] for u, v in bus_edges)
    total_length = sum(G[u][v]["length"] for u, v in bus_edges)
    elapsed = time.perf_counter() - t0

    return {
        "mode": "bus_brt",
        "criterion": criterion,
        "stations": stations,
        "subgraph": bus_subgraph,
        "edges": list(bus_edges),
        "total_cost": round(total_cost, 2),
        "total_length": round(total_length, 2),
        "execution_time_s": round(elapsed, 6),
    }


def design_subway_direct_mst(
    G: nx.Graph,
    regions: dict,
    criterion: str = "eccentricity",
    station_weight: str = "length",
    cost_per_meter: float = 120.0,
) -> dict:
    """
    Modo Metrô Subterrâneo (linha reta):
    1. Seleciona 1 estação por região (ex: grid 2x2).
    2. Ignora a malha de ruas e calcula a distância direta (linha reta) entre cada par de estações.
    3. Aplica Prim (MST) diretamente sobre essas arestas retas.
    """
    t0 = time.perf_counter()

    stations = {}
    for reg_id, nodes in sorted(regions.items()):
        if nodes:
            st, _ = find_best_station_in_region(G, nodes, criterion=criterion, weight=station_weight)
            if st is not None:
                stations[reg_id] = st

    station_list = list(dict.fromkeys(stations.values()))
    complete_graph = nx.Graph()

    for i in range(len(station_list)):
        u = station_list[i]
        for j in range(i + 1, len(station_list)):
            v = station_list[j]
            dist_m = _direct_distance_meters(G, u, v)
            complete_graph.add_edge(u, v, length=dist_m, cost=dist_m * cost_per_meter)

    mst_subway = nx.minimum_spanning_tree(complete_graph, weight="length", algorithm="prim")
    direct_edges = list(mst_subway.edges())
    total_length = sum(d["length"] for _, _, d in mst_subway.edges(data=True))
    total_cost = sum(d["cost"] for _, _, d in mst_subway.edges(data=True))
    elapsed = time.perf_counter() - t0

    return {
        "mode": "subway_direct",
        "criterion": criterion,
        "stations": stations,
        "mst_graph": mst_subway,
        "edges": direct_edges,
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
    """Calcula uma aproximação da Árvore de Steiner para conectar terminais."""
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
    """Compara rotas ótimas (Dijkstra) sob menor distância, menor tempo e menor custo."""
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
