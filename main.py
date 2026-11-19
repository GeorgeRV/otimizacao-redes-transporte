"""
Pipeline principal de execução: Modelagem e Otimização de Redes de Transporte Urbano.
Uso:
  python main.py                                         # Grafo sintético 12x12 (offline)
  python main.py --place "Copacabana, Rio de Janeiro, Brazil"
  python main.py --place "Francisco Badaró, Brazil"      # Cidades pequenas (usa fallback=40 km/h)
"""

import argparse
import pandas as pd
from src.data_loader import (
    create_synthetic_city,
    load_osm_network,
    assign_spatial_regions,
    export_to_csv,
)
from src.structural_analysis import compute_structural_metrics, compute_centralities
from src.optimization import (
    design_bus_network_mst,
    design_subway_direct_mst,
    compare_multicriteria_routes,
)
from src.robustness import targeted_attack_curve
from src.visualization import (
    plot_structural_vulnerabilities,
    plot_betweenness_heatmap,
    plot_bus_vs_subway,
)


def run_pipeline(place_name: str = None):
    print("=" * 75)
    print("MODELAGEM E OTIMIZAÇÃO DE REDES DE TRANSPORTE URBANO")
    print("=" * 75)

    # 1. Carregamento / Geração dos Dados
    if place_name:
        print(f"[1/4] Baixando rede real via OSMnx: {place_name} ...")
        G = load_osm_network(place_name, fallback_speed_kph=40.0, save_path="data/raw/city_osm.graphml")
    else:
        print("[1/4] Gerando malha urbana sintética (12x12) ...")
        G = create_synthetic_city(rows=12, cols=12, seed=42)

    # Divisões espaciais distintas: Metrô (2x2 = 4 regiões) e Ônibus/BRT (3x3 = 9 regiões)
    subway_regions = assign_spatial_regions(G, grid_rows=2, grid_cols=2, prefix="METRO", attr_name="region_metro")
    bus_regions = assign_spatial_regions(G, grid_rows=3, grid_cols=3, prefix="BUS", attr_name="region_bus")

    export_to_csv(G, "data/processed")
    print(f"      Grafo pronto: {G.number_of_nodes()} vértices | {G.number_of_edges()} arestas")
    print(f"      Regiões Metrô (2x2): {len(subway_regions)} | Regiões Ônibus (3x3): {len(bus_regions)}")

    # 2. Análise Estrutural
    print("\n[2/4] Executando análise estrutural e centralidades ...")
    metrics = compute_structural_metrics(G, weight="length")
    cent_df = compute_centralities(G, weight="travel_time")

    print(f"      Conectado: {metrics['is_connected']} | Pontes: {metrics['num_bridges']} | Pontos de Articulação: {metrics['num_articulation_points']}")
    print(f"      Eficiência Global: {metrics['global_efficiency']} | Distância Média: {metrics['avg_shortest_path_length']} m")

    # 3. Comparação de Modais (Ônibus 3x3 vs Metrô 2x2) e Critérios de Centralidade
    print("\n[3/4] Comparando critérios de centralidade para Ônibus/BRT (3x3) e Metrô (2x2) ...")
    comparison_rows = []
    bus_ecc = None
    subway_ecc = None

    for crit in ["eccentricity", "closeness", "betweenness"]:
        bus_res = design_bus_network_mst(G, bus_regions, criterion=crit)
        sub_res = design_subway_direct_mst(G, subway_regions, criterion=crit)
        if crit == "eccentricity":
            bus_ecc, subway_ecc = bus_res, sub_res

        comparison_rows.append({
            "modal": "Ônibus/BRT (3x3 - Ruas)",
            "criterio": crit,
            "estacoes": len(bus_res["stations"]),
            "extensao_m": bus_res["total_length"],
            "custo_estimado": bus_res["total_cost"],
        })
        comparison_rows.append({
            "modal": "Metrô (2x2 - Linha Reta)",
            "criterio": crit,
            "estacoes": len(sub_res["stations"]),
            "extensao_m": sub_res["total_length"],
            "custo_estimado": sub_res["total_cost"],
        })

    comp_df = pd.DataFrame(comparison_rows)
    print(comp_df.to_string(index=False))

    # 4. Análise de Robustez e Geração de Gráficos
    print("\n[4/4] Simulando ataques direcionados (robustez) e salvando gráficos ...")
    attack_df = targeted_attack_curve(G, k_steps=5, weight="travel_time")
    print(attack_df.to_string(index=False))

    plot_structural_vulnerabilities(G, metrics, save_path="docs/imagens/vulnerabilidades.png")
    plot_betweenness_heatmap(G, cent_df, save_path="docs/imagens/centralidade.png")
    plot_bus_vs_subway(G, bus_ecc, subway_ecc, save_path="docs/imagens/onibus_vs_metro.png")
    print("\nGráficos salvos em docs/imagens/ (incluindo onibus_vs_metro.png). Concluído!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Redes de Transporte Urbano")
    parser.add_argument("--place", type=str, default=None, help="Nome da cidade/bairro para baixar via OSMnx")
    args = parser.parse_args()
    run_pipeline(args.place)
