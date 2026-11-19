# Modelagem e Otimização de Redes de Transporte Urbano por meio da Teoria dos Grafos

**Autores:** George Rodrigues Vaz e Gerardo Mikael do Carmo Pereira

## Sobre o Projeto
Este repositório contém a implementação computacional em Python para modelagem, análise estrutural, otimização de infraestrutura e análise de robustez de redes viárias urbanas usando Teoria dos Grafos (`NetworkX` e `OSMnx`).

## Estrutura do Repositório
- `data/raw/`: Grafos brutos baixados do OpenStreetMap (`.graphml`).
- `data/processed/`: Nós e arestas processados em `.csv`.
- `src/`: Código-fonte modular em Python:
  - `data_loader.py`: Coleta de redes reais via OSMnx ou geração de malhas sintéticas.
  - `structural_analysis.py`: Conectividade, pontes, pontos de articulação e centralidades.
  - `optimization.py`: Seleção de estações por centro de subgrafo, Linha de Metrô (Prim/MST), Árvore de Steiner e Dijkstra multi-critério.
  - `robustness.py`: Simulação de falhas e ataques direcionados a gargalos.
  - `visualization.py`: Geração de gráficos em alta resolução (300 DPI) para o relatório em LaTeX.
- `notebooks/`: Cadernos Jupyter passo a passo (`01_coleta_dados`, `02_analise_estrutural`, `03_otimizacao_robustez`).
- `docs/`: Relatório em LaTeX e imagens exportadas.

## Como Executar
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute o pipeline completo com grafo sintético (offline):
   ```bash
   python main.py
   ```
3. Execute com uma rede viária real do OpenStreetMap:
   ```bash
   python main.py --place "Copacabana, Rio de Janeiro, Brazil"
   ```
