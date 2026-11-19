"""
Módulo de planejamento de infraestrutura de transporte urbano.
Utiliza a Centralidade de Excentricidade (Centro do subgrafo) para selecionar
estações com equidade espacial em cada região e Prim (MST) para conectar:
- Ônibus/BRT (grid 3x3): segue a malha viária real (Dijkstra + Prim).
- Metrô Subterrâneo (grid 2x2): liga estações em linha reta (Haversine/Euclidiana + Prim).
"""

import math
import time
import networkx as nx


def direct_distance_meters(G: nx.Graph, u, v) -> float:
    """Distância em linha reta (metros) via Haversine (OSM) ou Euclidiana (sintético)."""
    x1, y1 = float(G.nodes[u].get("x", 0.0)), float(G.nodes[u].get("y", 0.0))
    x2, y2 = float(G.nodes[v].get("x", 0.0)), float(G.nodes[v].get("y", 0.0))

    if G.graph.get("is_geographic", False):
        R = 6371000.0
        phi1, phi2 = math.radians(y1), math.radians(y2)
        dphi = math.radians(y2 - y1)
        dlambda = math.radians(x2 - x1)
        a = (
            math.sin(dphi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        )
        return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return math.hypot(x2 - x1, y2 - y1)


def find_region_center_station(G: nx.Graph, region_nodes: list, weight: str = "length"):
    """
    Encontra o centro da região (vértice de menor excentricidade / menor distância máxima
    até qualquer outro ponto da região), garantindo equidade de acesso.
    """
    subG = G.subgraph(region_nodes)
    if subG.number_of_nodes() == 0:
        return None, 0.0
    if not nx.is_connected(subG):
        largest = max(nx.connected_components(subG), key=len)
        subG = subG.subgraph(largest)

    best_station = None
    min_ecc = float("inf")
    for v in subG.nodes():
        lengths = nx.single_source_dijkstra_path_length(subG, v, weight=weight)
        ecc = max(lengths.values()) if lengths else 0.0
        if ecc < min_ecc:
            min_ecc = ecc
            best_station = v

    return best_station, round(min_ecc, 2)


def design_bus_network_mst(
    G: nx.Graph,
    regions: dict,
    station_weight: str = "length",
    infra_weight: str = "cost",
) -> dict:
    """Planeja o corredor de Ônibus/BRT na superfície (Dijkstra pelas ruas + Prim)."""
    t0 = time.perf_counter()
    stations = {}
    for reg_id, nodes in sorted(regions.items()):
        if nodes:
            st, _ = find_region_center_station(G, nodes, weight=station_weight)
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

    total_cost = sum(G[u][v][infra_weight] for u, v in bus_edges)
    total_length = sum(G[u][v]["length"] for u, v in bus_edges)

    return {
        "mode": "bus_brt",
        "stations": stations,
        "edges": list(bus_edges),
        "total_cost": round(total_cost, 2),
        "total_length": round(total_length, 2),
        "execution_time_s": round(time.perf_counter() - t0, 6),
    }


def design_subway_direct_mst(
    G: nx.Graph,
    regions: dict,
    station_weight: str = "length",
    cost_per_meter: float = 120.0,
) -> dict:
    """Planeja o Metrô Subterrâneo em linha reta (Distância direta + Prim)."""
    t0 = time.perf_counter()
    stations = {}
    for reg_id, nodes in sorted(regions.items()):
        if nodes:
            st, _ = find_region_center_station(G, nodes, weight=station_weight)
            if st is not None:
                stations[reg_id] = st

    station_list = list(dict.fromkeys(stations.values()))
    complete_graph = nx.Graph()

    for i in range(len(station_list)):
        u = station_list[i]
        for j in range(i + 1, len(station_list)):
            v = station_list[j]
            dist_m = direct_distance_meters(G, u, v)
            complete_graph.add_edge(u, v, length=dist_m, cost=dist_m * cost_per_meter)

    mst_subway = nx.minimum_spanning_tree(complete_graph, weight="length", algorithm="prim")
    direct_edges = list(mst_subway.edges())
    total_length = sum(d["length"] for _, _, d in mst_subway.edges(data=True))
    total_cost = sum(d["cost"] for _, _, d in mst_subway.edges(data=True))

    return {
        "mode": "subway_direct",
        "stations": stations,
        "mst_graph": mst_subway,
        "edges": direct_edges,
        "total_cost": round(total_cost, 2),
        "total_length": round(total_length, 2),
        "execution_time_s": round(time.perf_counter() - t0, 6),
    }
