"""
Pipeline principal de execução: Modelagem e Otimização de Redes de Transporte Urbano.
Uso:
  python main.py                  # Roda com grafo sintético (rápido, offline)
  python main.py --place "Copacabana, Rio de Janeiro, Brazil"   # Roda com dados reais do OpenStreetMap
"""

import argparse
from src.data_loader import (
    create_synthetic_city,
    load_osm_network,
    assign_spatial_regions,
    export_to_csv,
)
from src.structural_analysis import compute_structural_metrics, compute_centralities
from src.optimization import (
    design_subway_line_mst,
    compute_city_mst,
    compute_steiner_network,
    compare_multicriteria_routes,
)
from src.robustness import targeted_attack_curve
from src.visualization import (
    plot_structural_vulnerabilities,
    plot_betweenness_heatmap,
    plot_optimized_subway,
)


def run_pipeline(place_name: str = None):
    print("=" * 70)
    print("MODELAGEM E OTIMIZAÇÃO DE REDES DE TRANSPORTE URBANO")
    print("=" * 70)

    # 1. Carregamento / Geração dos Dados
    if place_name:
        print(f"[1/4] Baixando rede real via OSMnx: {place_name} ...")
        G = load_osm_network(place_name, save_path="data/raw/city_osm.graphml")
        regions = assign_spatial_regions(G, num_regions=4)
    else:
        print("[1/4] Gerando malha urbana sintética (8x6, 4 regiões) ...")
        G, regions = create_synthetic_city(rows=8, cols=6, num_regions=4, seed=42)

    export_to_csv(G, "data/processed")
    print(f"      Grafo pronto: {G.number_of_nodes()} vértices | {G.number_of_edges()} arestas")

    # 2. Análise Estrutural
    print("\n[2/4] Executando análise estrutural e centralidades ...")
    metrics = compute_structural_metrics(G, weight="length")
    cent_df = compute_centralities(G, weight="travel_time")

    print(f"      Conectado: {metrics['is_connected']} | Pontes: {metrics['num_bridges']} | Pontos de Articulação: {metrics['num_articulation_points']}")
    print(f"      Eficiência Global: {metrics['global_efficiency']} | Distância Média: {metrics['avg_shortest_path_length']} m")
    print("      Top 3 Gargalos (Betweenness Centrality):")
    print(cent_df[["node", "region", "betweenness_centrality", "closeness_centrality"]].head(3).to_string(index=False))

    # 3. Otimização de Infraestrutura (Metrô MST vs Árvore de Steiner) e Rotas
    print("\n[3/4] Otimizando infraestrutura e rotas ...")
    subway = design_subway_line_mst(G, regions, station_weight="length", infra_weight="cost")
    terminals = list(subway["stations"].values())
    steiner = compute_steiner_network(G, terminals, weight="cost")
    city_mst = compute_city_mst(G, weight="cost", algorithm="kruskal")

    print(f"      Estações selecionadas por região: {subway['stations']}")
    print(f"      Linha de Metrô (Prim): Custo = {subway['total_cost']} | Extensão = {subway['total_length']} m | Tempo = {subway['execution_time_s']}s")
    print(f"      Árvore de Steiner:     Custo = {steiner['total_cost']} | Extensão = {steiner['total_length']} m | Tempo = {steiner['execution_time_s']}s")
    print(f"      MST da Cidade Inteira: Custo = {city_mst['total_weight']} | Arestas = {city_mst['num_edges']}")

    if len(terminals) >= 2:
        routes = compare_multicriteria_routes(G, terminals[0], terminals[-1])
        print(f"\n      Comparação de Rotas ({terminals[0]} -> {terminals[-1]}):")
        for crit, info in routes.items():
            print(f"        - Otimizando {crit:11s}: {info['length_m']} m | {info['travel_time_s']} s | Custo {info['cost']}")

    # 4. Análise de Robustez e Geração de Gráficos
    print("\n[4/4] Simulando ataques direcionados (robustez) e salvando gráficos ...")
    attack_df = targeted_attack_curve(G, k_steps=5, weight="travel_time")
    print(attack_df.to_string(index=False))

    plot_structural_vulnerabilities(G, metrics, save_path="docs/imagens/vulnerabilidades.png")
    plot_betweenness_heatmap(G, cent_df, save_path="docs/imagens/centralidade.png")
    plot_optimized_subway(G, subway, steiner, save_path="docs/imagens/otimizacao_infra.png")
    print("\nGráficos salvos em docs/imagens/. Pipeline concluído com sucesso!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Redes de Transporte Urbano")
    parser.add_argument("--place", type=str, default=None, help="Nome da cidade/bairro para baixar via OSMnx")
    args = parser.parse_args()
    run_pipeline(args.place)
