"""
Pipeline principal: Modelagem, Planejamento de Infraestrutura e Roteamento Multimodal com Orçamento.
Uso:
  python main.py                                         # Malha sintética 12x12 (offline)
  python main.py --place "Copacabana, Rio de Janeiro, Brazil"
  python main.py --place "Francisco Badaró, Brazil"      # Cidades pequenas (fallback=40 km/h)
"""

import argparse
import pandas as pd
from src.data_loader import (
    create_synthetic_city,
    load_osm_network,
    assign_spatial_regions,
    export_to_csv,
)
from src.structural_analysis import compute_network_summary, compute_centralities
from src.optimization import design_bus_network_mst, design_subway_direct_mst
from src.multimodal_routing import build_multimodal_graph, budget_constrained_dijkstra
from src.visualization import plot_bus_vs_subway, plot_budget_routes_comparison, plot_betweenness_heatmap


def pick_distant_pair(G):
    """Escolhe automaticamente um par (origem, destino) em extremos opostos do mapa."""
    nodes = list(G.nodes())
    source = min(nodes, key=lambda n: G.nodes[n]["x"] + G.nodes[n]["y"])
    target = max(nodes, key=lambda n: G.nodes[n]["x"] + G.nodes[n]["y"])
    return source, target


def run_pipeline(place_name: str = None):
    print("=" * 75)
    print("MODELAGEM DE TRANSPORTE URBANO E ROTEAMENTO MULTIMODAL POR ORÇAMENTO")
    print("=" * 75)

    # 1. Carregamento dos Dados
    if place_name:
        print(f"[1/4] Baixando rede real via OSMnx: {place_name} ...")
        G = load_osm_network(place_name, fallback_speed_kph=40.0, save_path="data/raw/city_osm.graphml")
    else:
        print("[1/4] Gerando malha urbana sintética (12x12) ...")
        G = create_synthetic_city(rows=12, cols=12, seed=42)

    subway_regions = assign_spatial_regions(G, grid_rows=2, grid_cols=2, prefix="METRO", attr_name="region_metro")
    bus_regions = assign_spatial_regions(G, grid_rows=3, grid_cols=3, prefix="BUS", attr_name="region_bus")
    export_to_csv(G, "data/processed")

    # 2. Diagnóstico Estrutural Breve
    summary = compute_network_summary(G)
    cent_df = compute_centralities(G, weight='travel_time')
    print("\n[2/4] Diagnóstico estrutural da malha viária:")
    print(f"      Vértices: {summary['num_nodes']} | Arestas: {summary['num_edges']} | Conectado: {summary['is_connected']}")
    print(f"      Pontes: {summary['num_bridges']} | Pontos de Articulação: {summary['num_articulation_points']} | Distância Média: {summary['avg_shortest_path_m']} m")

    # 3. Planejamento de Infraestrutura (Critério: Excentricidade + Prim)
    print("\n[3/4] Planejando infraestrutura (Centro por Excentricidade + Prim):")
    bus_res = design_bus_network_mst(G, bus_regions)
    subway_res = design_subway_direct_mst(G, subway_regions)

    print(f"      Ônibus/BRT (3x3 - Ruas):      {len(bus_res['stations'])} paradas | Extensão: {bus_res['total_length']} m | Custo: {bus_res['total_cost']}")
    print(f"      Metrô (2x2 - Linha Reta):     {len(subway_res['stations'])} estações | Extensão: {subway_res['total_length']} m | Custo: {subway_res['total_cost']}")

    # 4. Roteamento Multimodal com Restrição de Orçamento
    print("\n[4/4] Guia de Rotas Multimodal com Restrição de Orçamento (Dijkstra):")
    M = build_multimodal_graph(G, bus_res, subway_res)
    source, target = pick_distant_pair(G)
    budgets = [0.0, 10.0, 20.0, 40.0]

    routes_list = []
    rows = []
    for b in budgets:
        res = budget_constrained_dijkstra(M, source, target, max_budget=b)
        routes_list.append(res)
        rows.append({
            "orcamento_max_R$": b,
            "gasto_R$": res["total_fare_brl"],
            "tempo_min": res["total_time_min"],
            "distancia_m": res["total_length_m"],
            "sequencia_modais": res["modes_sequence"],
        })

    print(f"      Trajeto analisado: Nó {source} -> Nó {target}")
    print(pd.DataFrame(rows).to_string(index=False))

    # Gráficos para o LaTeX
    plot_betweenness_heatmap(G, cent_df, save_path="docs/imagens/centralidade.png")
    plot_bus_vs_subway(G, bus_res, subway_res, save_path="docs/imagens/onibus_vs_metro.png")
    plot_budget_routes_comparison(G, routes_list, source, target, save_path="docs/imagens/rotas_orcamento.png")
    print("\nGráficos salvos em docs/imagens/ (centralidade.png, onibus_vs_metro.png e rotas_orcamento.png).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Redes de Transporte Urbano")
    parser.add_argument("--place", type=str, default=None, help="Nome da cidade/bairro para baixar via OSMnx")
    args = parser.parse_args()
    run_pipeline(args.place)
