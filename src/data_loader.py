"""
Módulo de coleta, geração e carregamento de redes viárias urbanas.
Suporta tanto redes reais do OpenStreetMap (via OSMnx) quanto grafos sintéticos
em malha, com particionamento espacial flexível (ex: 2x2 para Metrô, 3x3 para Ônibus).
"""

import os
import math
import random
import pandas as pd
import networkx as nx

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False


def load_osm_network(
    place_name: str,
    network_type: str = "drive",
    fallback_speed_kph: float = 40.0,
    save_path: str = None,
) -> nx.Graph:
    """
    Baixa a rede viária real do OpenStreetMap usando OSMnx e a converte em um
    grafo não-direcionado simples com atributos padronizados (length, speed_limit, travel_time, cost).
    Usa `fallback_speed_kph` para cidades pequenas sem tags 'maxspeed' cadastradas.
    """
    if not OSMNX_AVAILABLE:
        raise ImportError(
            "A biblioteca 'osmnx' não está instalada. Instale com: pip install osmnx"
        )

    G_multi = ox.graph_from_place(place_name, network_type=network_type)
    G_multi = ox.add_edge_speeds(G_multi, fallback=fallback_speed_kph)
    G_multi = ox.add_edge_travel_times(G_multi)

    G = nx.Graph(is_geographic=True)
    for node, data in G_multi.nodes(data=True):
        G.add_node(
            node,
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            region="OSM",
            type="intersection",
        )

    for u, v, data in G_multi.edges(data=True):
        if u == v:
            continue
        length = float(data.get("length", 100.0))
        speed_kph = float(data.get("speed_kph", fallback_speed_kph))
        travel_time = float(data.get("travel_time", length / (speed_kph * 1000.0 / 3600.0)))
        cost = round(length * 1.5, 2)

        if G.has_edge(u, v):
            if length < G[u][v]["length"]:
                G[u][v].update(
                    length=length,
                    speed_limit=speed_kph,
                    travel_time=travel_time,
                    cost=cost,
                )
        else:
            G.add_edge(
                u,
                v,
                length=length,
                speed_limit=speed_kph,
                travel_time=travel_time,
                cost=cost,
            )

    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    G.graph["is_geographic"] = True

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        nx.write_graphml(G, save_path)

    return G


def assign_spatial_regions(
    G: nx.Graph,
    grid_rows: int = 2,
    grid_cols: int = 2,
    prefix: str = "REG",
    attr_name: str = "region",
) -> dict:
    """
    Particiona os nós do grafo em um grid espacial de tamanho (grid_rows x grid_cols)
    usando as coordenadas (x, y). Permite criar divisões diferentes no mesmo mapa
    (ex: 2x2 para Metrô e 3x3 para Ônibus/BRT).
    """
    nodes = list(G.nodes())
    xs = [float(G.nodes[n].get("x", 0.0)) for n in nodes]
    ys = [float(G.nodes[n].get("y", 0.0)) for n in nodes]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    regions = {}
    for n in nodes:
        x = float(G.nodes[n].get("x", 0.0))
        y = float(G.nodes[n].get("y", 0.0))
        col = min(grid_cols - 1, int((x - min_x) / (max_x - min_x + 1e-9) * grid_cols))
        row = min(grid_rows - 1, int((y - min_y) / (max_y - min_y + 1e-9) * grid_rows))
        reg_id = f"{prefix}-{row * grid_cols + col:02d}"
        G.nodes[n][attr_name] = reg_id
        regions.setdefault(reg_id, []).append(n)

    return regions


def create_synthetic_city(
    rows: int = 12,
    cols: int = 12,
    drop_edge_prob: float = 0.08,
    seed: int = 42,
) -> nx.Graph:
    """
    Gera um grafo sintético representando uma malha urbana com coordenadas (x, y)
    em metros (espaçamento médio de 150m por quarteirão) e atributos de distância,
    velocidade, tempo e custo.
    """
    rng = random.Random(seed)
    G_grid = nx.grid_2d_graph(rows, cols)
    G = nx.Graph(is_geographic=False)

    node_map = {coord: idx for idx, coord in enumerate(G_grid.nodes())}
    node_types = ["residential", "commercial", "industrial", "tourist"]

    for (r, c), idx in node_map.items():
        G.add_node(
            idx,
            x=float(c * 150.0),
            y=float((rows - 1 - r) * 150.0),
            region="SYN",
            type=rng.choice(node_types),
        )

    for u_coord, v_coord in G_grid.edges():
        u, v = node_map[u_coord], node_map[v_coord]
        length = float(rng.randint(120, 220))
        speed_limit = float(rng.choice([30, 40, 50, 60]))
        travel_time = round(length / (speed_limit * 1000.0 / 3600.0), 2)
        cost = round(length * rng.uniform(10.0, 25.0), 2)
        G.add_edge(
            u,
            v,
            length=length,
            speed_limit=speed_limit,
            travel_time=travel_time,
            cost=cost,
        )

    edges = list(G.edges())
    rng.shuffle(edges)
    for u, v in edges:
        if rng.random() < drop_edge_prob:
            edge_data = G[u][v].copy()
            G.remove_edge(u, v)
            if not nx.is_connected(G):
                G.add_edge(u, v, **edge_data)

    return G


def export_to_csv(G: nx.Graph, output_dir: str) -> None:
    """Exporta nós e arestas do grafo para arquivos CSV padronizados."""
    os.makedirs(output_dir, exist_ok=True)
    nodes_df = pd.DataFrame([{"id": n, **data} for n, data in G.nodes(data=True)])
    edges_df = pd.DataFrame(
        [{"source": u, "target": v, **data} for u, v, data in G.edges(data=True)]
    )
    nodes_df.to_csv(os.path.join(output_dir, "nodes.csv"), index=False)
    edges_df.to_csv(os.path.join(output_dir, "edges.csv"), index=False)


def load_from_csv(nodes_path: str, edges_path: str, is_geographic: bool = False) -> nx.Graph:
    """Carrega um grafo salvo em CSV."""
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    G = nx.Graph(is_geographic=is_geographic)
    id_col = "id" if "id" in nodes_df.columns else "node"

    for _, row in nodes_df.iterrows():
        n_id = row[id_col]
        attrs = row.drop(labels=[id_col]).to_dict()
        G.add_node(n_id, **attrs)

    for _, row in edges_df.iterrows():
        u, v = row["source"], row["target"]
        attrs = row.drop(labels=["source", "target"]).to_dict()
        length = float(attrs.get("length", 100.0))
        speed = float(attrs.get("speed_limit", 40.0))
        if "travel_time" not in attrs:
            attrs["travel_time"] = round(length / (speed * 1000.0 / 3600.0), 2)
        G.add_edge(u, v, **attrs)

    return G
