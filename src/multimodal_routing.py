"""
Módulo de Roteamento Multimodal com Restrição de Orçamento (Resource-Constrained Shortest Path).

Constrói um grafo direcionado em camadas (multilayer graph) integrando:
  1. 'walk'   (A pé): todas as ruas, R$ 0,00, 5 km/h.
  2. 'bus'    (Ônibus/BRT): corredor de ônibus (grid 3x3), tarifa fixa de embarque (ex: R$ 5,00), 22 km/h.
  3. 'subway' (Metrô): ligações em linha reta (grid 2x2), tarifa fixa de embarque (ex: R$ 12,00), 55 km/h.
  4. 'car'    (Carro/App): todas as ruas, bandeirada (ex: R$ 6,00) + tarifa por km (ex: R$ 4,00/km), 35 km/h.

O algoritmo encontra a rota de MENOR TEMPO entre origem e destino respeitando
um orçamento máximo (em R$) definido pelo passageiro, usando uma extensão do
algoritmo de Dijkstra com poda por dominância de Pareto.
"""

import heapq
import networkx as nx
from src.optimization import direct_distance_meters


DEFAULT_MODES_CONFIG = {
    "walk": {"speed_kph": 5.0, "boarding_fare": 0.0, "fare_per_km": 0.0, "transfer_time_s": 0.0},
    "bus": {"speed_kph": 22.0, "boarding_fare": 5.0, "fare_per_km": 0.0, "transfer_time_s": 180.0},
    "subway": {"speed_kph": 55.0, "boarding_fare": 12.0, "fare_per_km": 0.0, "transfer_time_s": 240.0},
    "car": {"speed_kph": 35.0, "boarding_fare": 6.0, "fare_per_km": 4.0, "transfer_time_s": 120.0},
}


def build_multimodal_graph(
    G: nx.Graph,
    bus_result: dict,
    subway_result: dict,
    config: dict = None,
) -> nx.DiGraph:
    """
    Constrói o grafo multimodal em camadas onde cada vértice é um par (node_id, camada).
    Arestas intra-camada representam deslocamento no modal; arestas inter-camada representam
    embarque/desembarque (com tarifa de embarque e tempo de espera).
    """
    cfg = config or DEFAULT_MODES_CONFIG
    M = nx.DiGraph()

    def _time_s(length_m: float, speed_kph: float) -> float:
        return length_m / (speed_kph * 1000.0 / 3600.0)

    # 1. Camada A PÉ (todas as ruas) e Camada CARRO/APP (todas as ruas)
    for u, v, data in G.edges(data=True):
        length = float(data.get("length", 100.0))

        # A pé
        t_walk = _time_s(length, cfg["walk"]["speed_kph"])
        M.add_edge((u, "walk"), (v, "walk"), time=t_walk, fare=0.0, length=length, mode="walk")
        M.add_edge((v, "walk"), (u, "walk"), time=t_walk, fare=0.0, length=length, mode="walk")

        # Carro/App
        t_car = _time_s(length, cfg["car"]["speed_kph"])
        f_car = (length / 1000.0) * cfg["car"]["fare_per_km"]
        M.add_edge((u, "car"), (v, "car"), time=t_car, fare=f_car, length=length, mode="car")
        M.add_edge((v, "car"), (u, "car"), time=t_car, fare=f_car, length=length, mode="car")

    # Transferências entre a pé e carro (disponível em qualquer cruzamento)
    for n in G.nodes():
        M.add_edge(
            (n, "walk"), (n, "car"),
            time=cfg["car"]["transfer_time_s"], fare=cfg["car"]["boarding_fare"],
            length=0.0, mode="transfer"
        )
        M.add_edge((n, "car"), (n, "walk"), time=30.0, fare=0.0, length=0.0, mode="transfer")

    # 2. Camada ÔNIBUS/BRT (arestas do corredor de ônibus)
    bus_nodes = set()
    for u, v in bus_result["edges"]:
        length = float(G[u][v].get("length", 100.0))
        t_bus = _time_s(length, cfg["bus"]["speed_kph"])
        M.add_edge((u, "bus"), (v, "bus"), time=t_bus, fare=0.0, length=length, mode="bus")
        M.add_edge((v, "bus"), (u, "bus"), time=t_bus, fare=0.0, length=length, mode="bus")
        bus_nodes.update([u, v])

    # Embarque/desembarque nos nós do corredor de ônibus
    for n in bus_nodes:
        M.add_edge(
            (n, "walk"), (n, "bus"),
            time=cfg["bus"]["transfer_time_s"], fare=cfg["bus"]["boarding_fare"],
            length=0.0, mode="transfer"
        )
        M.add_edge((n, "bus"), (n, "walk"), time=30.0, fare=0.0, length=0.0, mode="transfer")

    # 3. Camada METRÔ SUBTERRÂNEO (arestas em linha reta)
    subway_nodes = set()
    for u, v in subway_result["edges"]:
        length = direct_distance_meters(G, u, v)
        t_sub = _time_s(length, cfg["subway"]["speed_kph"])
        M.add_edge((u, "subway"), (v, "subway"), time=t_sub, fare=0.0, length=length, mode="subway")
        M.add_edge((v, "subway"), (u, "subway"), time=t_sub, fare=0.0, length=length, mode="subway")
        subway_nodes.update([u, v])

    # Embarque/desembarque apenas nas estações de metrô
    for n in subway_nodes:
        M.add_edge(
            (n, "walk"), (n, "subway"),
            time=cfg["subway"]["transfer_time_s"], fare=cfg["subway"]["boarding_fare"],
            length=0.0, mode="transfer"
        )
        M.add_edge((n, "subway"), (n, "walk"), time=60.0, fare=0.0, length=0.0, mode="transfer")

    return M


def budget_constrained_dijkstra(
    M: nx.DiGraph,
    source_node,
    target_node,
    max_budget: float,
) -> dict:
    """
    Dijkstra com Restrição de Orçamento (Resource-Constrained Shortest Path).
    Encontra o caminho de menor tempo de (source_node, 'walk') até (target_node, 'walk')
    com custo monetário acumulado <= max_budget.
    """
    start = (source_node, "walk")
    goal = (target_node, "walk")

    # Fila de prioridade: (tempo_acumulado, custo_acumulado, estado_atual, caminho)
    pq = [(0.0, 0.0, start, [start])]

    # Poda por dominância: para cada estado, guarda lista de pares (custo, tempo) não-dominados
    best_labels = {}

    while pq:
        curr_time, curr_fare, state, path = heapq.heappop(pq)

        if state == goal:
            # Extrair segmentos por modal
            segments = []
            modes_used = []
            total_length = 0.0
            for i in range(len(path) - 1):
                u_s, v_s = path[i], path[i + 1]
                edata = M[u_s][v_s]
                mode = edata["mode"]
                total_length += edata["length"]
                if mode != "transfer":
                    segments.append((u_s[0], v_s[0], mode))
                    if not modes_used or modes_used[-1] != mode:
                        modes_used.append(mode)

            return {
                "found": True,
                "budget_limit": max_budget,
                "total_time_min": round(curr_time / 60.0, 2),
                "total_fare_brl": round(curr_fare, 2),
                "total_length_m": round(total_length, 1),
                "modes_sequence": " -> ".join(modes_used),
                "segments": segments,
                "path": path,
            }

        # Verifica se já existe um rótulo que domina (curr_fare, curr_time)
        labels = best_labels.get(state, [])
        if any(f <= curr_fare + 1e-6 and t <= curr_time + 1e-6 for f, t in labels):
            continue

        # Remove rótulos antigos dominados pelo novo e registra o atual
        labels = [(f, t) for f, t in labels if not (curr_fare <= f and curr_time <= t)]
        labels.append((curr_fare, curr_time))
        best_labels[state] = labels

        for nxt in M.neighbors(state):
            edata = M[state][nxt]
            new_fare = round(curr_fare + edata["fare"], 2)
            if new_fare <= max_budget + 1e-6:
                new_time = curr_time + edata["time"]
                heapq.heappush(pq, (new_time, new_fare, nxt, path + [nxt]))

    return {
        "found": False,
        "budget_limit": max_budget,
        "total_time_min": None,
        "total_fare_brl": None,
        "total_length_m": None,
        "modes_sequence": "Sem rota",
        "segments": [],
        "path": [],
    }
