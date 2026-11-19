"""
Módulo de coleta, geração e carregamento de redes viárias urbanas.
Suporta tanto redes reais do OpenStreetMap (via OSMnx) quanto grafos sintéticos
em malha (compatíveis com a estrutura de regiões do projeto original).
"""

import os
import random
import pandas as pd
import networkx as nx

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False


def load_osm_network(place_name: str, network_type: str = "drive", save_path: str = None) -> nx.Graph:
    """
    Baixa a rede viária real do OpenStreetMap usando OSMnx e a converte em um
    grafo não-direcionado simples com atributos padronizados (length, speed_limit, travel_time, cost).
    """
    if not OSMNX_AVAILABLE:
        raise ImportError(
            "A biblioteca 'osmnx' não está instalada. Instale com: pip install osmnx"
        )

    G_multi = ox.graph_from_place(place_name, network_type=network_type)
    G_multi = ox.add_edge_speeds(G_multi)
    G_multi = ox.add_edge_travel_times(G_multi)

    # Converte para grafo não-direcionado simples mantendo a aresta de menor distância
    G = nx.Graph()
    for node, data in G_multi.nodes(data=True):
        G.add_node(
            node,
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            region="OSM",
            type="intersection",
        )

    for u, v, data in G_multi.edges(data=True):
        if u == v:
            continue
        length = float(data.get("length", 100.0))
        speed_kph = float(data.get("speed_kph", 40.0))
        travel_time = float(data.get("travel_time", length / (speed_kph * 1000 / 3600)))
        # Custo de infraestrutura proporcional à distância (pode ser customizado por tipo de via)
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

    # Mantém a maior componente conexa para análises de conectividade/rotas
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        nx.write_graphml(G, save_path)

    return G


def assign_spatial_regions(G: nx.Graph, num_regions: int = 4) -> dict:
    """
    Particiona os nós do grafo em regiões geográficas (quadrantes/setores) com base
    nas coordenadas (x, y), útil para aplicar algoritmos de estações por região em dados reais.
    """
    nodes = list(G.nodes())
    xs = [G.nodes[n].get("x", 0.0) for n in nodes]
    ys = [G.nodes[n].get("y", 0.0) for n in nodes]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    grid_side = max(1, int(num_regions ** 0.5))
    regions = {}

    for n in nodes:
        x = G.nodes[n].get("x", 0.0)
        y = G.nodes[n].get("y", 0.0)
        col = min(grid_side - 1, int((x - min_x) / (max_x - min_x + 1e-9) * grid_side))
        row = min(grid_side - 1, int((y - min_y) / (max_y - min_y + 1e-9) * grid_side))
        reg_id = f"REG-{row * grid_side + col:02d}"
        G.nodes[n]["region"] = reg_id
        regions.setdefault(reg_id, []).append(n)

    return regions


def create_synthetic_city(
    rows: int = 8,
    cols: int = 6,
    num_regions: int = 4,
    drop_edge_prob: float = 0.08,
    seed: int = 42,
) -> tuple:
    """
    Gera um grafo sintético representando uma malha urbana com regiões (CEPs),
    coordenadas espaciais (x, y) e atributos de distância, velocidade, tempo e custo.
    A remoção probabilística de arestas (mantendo conectividade) cria pontes e
    pontos de articulação realistas para análise estrutural.
    """
    rng = random.Random(seed)
    G_grid = nx.grid_2d_graph(rows, cols)
    G = nx.Graph()

    node_map = {coord: idx for idx, coord in enumerate(G_grid.nodes())}
    total_nodes = rows * cols
    nodes_per_region = total_nodes // num_regions
    extra_nodes = total_nodes % num_regions

    regions = {}
    start = 0
    for i in range(num_regions):
        size = nodes_per_region + (1 if i < extra_nodes else 0)
        reg_id = f"CEP-{10000 + i}"
        regions[reg_id] = list(range(start, start + size))
        start += size

    node_to_region = {n: reg for reg, nlist in regions.items() for n in nlist}
    node_types = ["residential", "commercial", "industrial", "tourist"]

    for (r, c), idx in node_map.items():
        G.add_node(
            idx,
            x=float(c),
            y=float(rows - 1 - r),
            region=node_to_region[idx],
            type=rng.choice(node_types),
        )

    for u_coord, v_coord in G_grid.edges():
        u, v = node_map[u_coord], node_map[v_coord]
        length = float(rng.randint(80, 250))  # metros
        speed_limit = float(rng.choice([30, 40, 50, 60]))  # km/h
        travel_time = round(length / (speed_limit * 1000.0 / 3600.0), 2)  # segundos
        cost = round(length * rng.uniform(10.0, 25.0), 2)  # custo de obra
        G.add_edge(
            u,
            v,
            length=length,
            speed_limit=speed_limit,
            travel_time=travel_time,
            cost=cost,
        )

    # Remove algumas arestas aleatoriamente sem desconectar o grafo (gera cenários mais realistas)
    edges = list(G.edges())
    rng.shuffle(edges)
    for u, v in edges:
        if rng.random() < drop_edge_prob:
            edge_data = G[u][v].copy()
            G.remove_edge(u, v)
            if not nx.is_connected(G):
                G.add_edge(u, v, **edge_data)

    return G, regions


def export_to_csv(G: nx.Graph, output_dir: str) -> None:
    """Exporta nós e arestas do grafo para arquivos CSV padronizados."""
    os.makedirs(output_dir, exist_ok=True)
    nodes_df = pd.DataFrame(
        [{"id": n, **data} for n, data in G.nodes(data=True)]
    )
    edges_df = pd.DataFrame(
        [{"source": u, "target": v, **data} for u, v, data in G.edges(data=True)]
    )
    nodes_df.to_csv(os.path.join(output_dir, "nodes.csv"), index=False)
    edges_df.to_csv(os.path.join(output_dir, "edges.csv"), index=False)


def load_from_csv(nodes_path: str, edges_path: str) -> tuple:
    """Carrega um grafo salvo em CSV (compatível com os arquivos do projeto antigo)."""
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    G = nx.Graph()
    regions = {}
    id_col = "id" if "id" in nodes_df.columns else "node"

    for _, row in nodes_df.iterrows():
        n_id = row[id_col]
        attrs = row.drop(labels=[id_col]).to_dict()
        G.add_node(n_id, **attrs)
        reg = attrs.get("region", "DEFAULT")
        regions.setdefault(reg, []).append(n_id)

    for _, row in edges_df.iterrows():
        u, v = row["source"], row["target"]
        attrs = row.drop(labels=["source", "target"]).to_dict()
        length = float(attrs.get("length", 100.0))
        speed = float(attrs.get("speed_limit", 40.0))
        if "travel_time" not in attrs:
            attrs["travel_time"] = round(length / (speed * 1000.0 / 3600.0), 2)
        G.add_edge(u, v, **attrs)

    return G, regions
