# Modelagem de Transporte Urbano e Roteamento Multimodal com Restrição de Orçamento

**Autores:** George Rodrigues Vaz e Gerardo Mikael do Carmo Pereira

## Estrutura do Projeto
- `src/data_loader.py`: Coleta de redes reais via OSMnx (`fallback=40 km/h`) ou geração de malha sintética, com particionamento espacial em grids distintos (2×2 para Metrô e 3×3 para Ônibus/BRT).
- `src/structural_analysis.py`: Diagnóstico estrutural breve da malha viária (conectividade, pontes, pontos de articulação e distância média).
- `src/optimization.py`: Seleção de estações por **Centralidade de Excentricidade** (equidade espacial) e conexão via **Prim (MST)** — na superfície para Ônibus/BRT e em linha reta para Metrô Subterrâneo.
- `src/multimodal_routing.py`: Construção do **Grafo Multimodal em Camadas** (a pé, ônibus, metrô e carro/app) e algoritmo **Dijkstra com Restrição de Orçamento**.
- `src/visualization.py`: Geração de gráficos em 300 DPI prontos para o relatório em LaTeX.

## Como Executar
```bash
pip install -r requirements.txt
python main.py                                     # Teste rápido offline (malha 12x12)
python main.py --place "Francisco Badaró, Brazil"  # Cidade pequena
python main.py --place "Copacabana, Rio de Janeiro, Brazil"
```
