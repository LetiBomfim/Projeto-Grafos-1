#>> Projeto 1 de Grafos 
#>> Disciplina: Teoria dos Grafos — UnB
#>> Turma: 01, 2025/2
#>> Professor: Dibio
#>>> Integrantes:
#>- Julia Paulo Amorim - 241039270
#>- Leticia Gonçalves Bomfim - 241002411
#>- Vitor Alencar Ribeiro - 231036292


# O relatorio se encontra em relatorio.ipynb
# O link da página do repositório é: https://github.com/LetiBomfim/Projeto-Grafos-1
# O link para clone do repositório é: https://github.com/LetiBomfim/Projeto-Grafos-1.git

import networkx as nx
import requests
import numpy as np
import matplotlib.pyplot as plt
import community as community_louvain
from matplotlib.patches import Patch


class FacebookGraph:
    def __init__(self):
        self.G = None
        self.G_subset = None

    def baixar_dados(self):

        url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"

        try:
            #Requests é uma biblioteca cliente HTTP para Python
            response = requests.get(url)
            response.raise_for_status()

            #Abre o arquivo com as arestas
            with open("facebook_combined.txt.gz", "wb") as f:
                f.write(response.content)
            
            print("Arquivo aberto com sucesso!\n")
            
            return True
        
        except Exception as e:
            print(f"Não foi possível abrir o arquivo: {e}")
            return False
    
    def carregar_rede(self):
        try:
            #Função nativa do NetworkX que lê um arquivo de lista de arestas e cria um grafo
            self.G = nx.read_edgelist("facebook_combined.txt.gz")

            #Teste se criou o grafo corretamente
            print(f" - Nós: {self.G.number_of_nodes()}")          #4039
            print(f" - Arestas: {self.G.number_of_edges()}")      #88234
            print(f" - Densidade: {nx.density(self.G):.6f}")      #0.010820

            return True
        
        except Exception as e:
            print(f"Erro ao criar grafo: {e}")
            return False
    
    def extrair_subconjunto(self, n_nos=2000):
        if self.G is None:
            print("O grafo não existe!")
            return False
        
        todos_nos = list(self.G.nodes())
        #Utiliza a biblioteca numpy para escolher nós aleatórios dentre os da lista
        #replace=False é para não haver repetição de nós
        nos_selecionados = np.random.choice(todos_nos, size=n_nos, replace=False)

        #G_subset é o subgrafo com os 2000 selecionados
        self.G_subset = self.G.subgraph(nos_selecionados).copy()

        for node in nos_selecionados:
            vizinhos = list(self.G.neighbors(node))
            for vizinho in vizinhos:
                if vizinho in nos_selecionados:
                    self.G_subset.add_edge(node, vizinho)

        print("\nInformações gerais dos 2000 nós selecionados:")
        print(f" - Nós: {self.G_subset.number_of_nodes()}")
        print(f" - Arestas: {self.G_subset.number_of_edges()}")
        print(f" - Densidade: {nx.density(self.G_subset):.6f}")

        return True
    
    def calcular_metricas(self, top_n=5):
        #Calcula as métricas de centralidade e detecta comunidades no subgrafo.
        if self.G_subset is None:
            print("O subgrafo não existe! Execute extrair_subconjunto() primeiro.")
            return False
        
        print("\n--- CALCULANDO MÉTRICAS DE CENTRALIDADE E COMUNIDADES ---")
        
        try:
            #Degree Centrality
            print("\n[1] Grau de Centralidade (Degree):")
            print("Mede a popularidade de um nó pelo número de conexões diretas.")
            degree_centrality = nx.degree_centrality(self.G_subset)
            sorted_degree = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)
            print(f"  Top {top_n} nós com maior Grau de Centralidade:")
            for i in range(top_n):
                print(f"    {i+1}. Nó {sorted_degree[i][0]}: {sorted_degree[i][1]:.4f}")

            #Betweenness Centrality
            print("\n[2] Centralidade de Intermediação (Betweenness):")
            print("Mede a importância de um nó como 'ponte' nos caminhos mais curtos entre outros nós.")
            betweenness_centrality = nx.betweenness_centrality(self.G_subset)
            sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True)
            print(f"  Top {top_n} nós com maior Centralidade de Intermediação:")
            for i in range(top_n):
                print(f"    {i+1}. Nó {sorted_betweenness[i][0]}: {sorted_betweenness[i][1]:.4f}")

            #Closeness Centrality
            print("\n[3] Centralidade de Proximidade (Closeness):")
            print("Mede quão rápido um nó consegue alcançar todos os outros na rede.")
            closeness_centrality = nx.closeness_centrality(self.G_subset)
            sorted_closeness = sorted(closeness_centrality.items(), key=lambda item: item[1], reverse=True)
            print(f"  Top {top_n} nós com maior Centralidade de Proximidade:")
            for i in range(top_n):
                print(f"    {i+1}. Nó {sorted_closeness[i][0]}: {sorted_closeness[i][1]:.4f}")

            #Eigenvector Centrality
            print("\n[4] Centralidade de Autovetor (Eigenvector):")
            print("Mede a influência de um nó com base na importância de seus vizinhos.")
            try:
                eigenvector_centrality = nx.eigenvector_centrality(self.G_subset, max_iter=1000)
                sorted_eigenvector = sorted(eigenvector_centrality.items(), key=lambda item: item[1], reverse=True)
                print(f"  Top {top_n} nós com maior Centralidade de Autovetor:")
                for i in range(top_n):
                    print(f"    {i+1}. Nó {sorted_eigenvector[i][0]}: {sorted_eigenvector[i][1]:.4f}")
            except nx.PowerIterationFailedConvergence:
                print("  Cálculo de autovetor não convergiu. Pulando esta métrica.")

            #Algoritmo de Louvain
            print("\n[5] Mapeamento de Comunidades (Louvain):")
            print("Agrupa os nós em 'panelinhas' onde as conexões internas são mais fortes.")
            communities = community_louvain.best_partition(self.G_subset)
            num_communities = len(set(communities.values()))
            print(f"  Número de comunidades detectadas: {num_communities}")

            # Armazenando os resultados na classe para uso posterior
            self.centrality_measures = {
                'degree': degree_centrality,
                'betweenness': betweenness_centrality,
                'closeness': closeness_centrality,
                'eigenvector': eigenvector_centrality if 'eigenvector_centrality' in locals() else None
            }
            self.communities = communities
            
            return True
            
        except Exception as e:
            print(f"Erro ao calcular as métricas: {e}")
            return False
    
    def visualizar_rede(self):
        try:
            if self.G_subset is None:
                print("O subgrafo não existe!")
                return False
            
            plt.figure(figsize=(20, 15))

            pos = nx.spring_layout(self.G_subset, k=2, iterations=1000)

            #Calcula o grau de cada nó coloca em um dicionário e o tamanho do nó é proporcional ao grau
            graus = dict(self.G_subset.degree())
            tamanhos = [np.log(graus[node]+1)*25 for node in self.G_subset.nodes()]

            #Para melhor visualização no grafo, os nós tem cores de acordo com o grau
            graus_valores = list(graus.values())
            percentis = np.percentile(graus_valores, [25, 50, 75, 95])

            node_colors = []
            for node in self.G_subset.nodes():
                if graus[node] <= percentis[0]:
                    node_colors.append(0)
                elif graus[node] <= percentis[1]:
                    node_colors.append(1)
                elif graus[node] <= percentis[2]:
                    node_colors.append(2)
                elif graus[node] <= percentis[3]:
                    node_colors.append(3)
                else:
                    node_colors.append(4)
            
            #Definições de visualização
            nx.draw_networkx_nodes(self.G_subset, pos, node_size=tamanhos, node_color=node_colors, cmap="plasma", alpha=1)
            nx.draw_networkx_edges(self.G_subset, pos, alpha=1, edge_color="gray", width=0.5)

            #Definição da legenda
            legend_labels = [f"Grau ≤ {int(percentis[0])}", f"Grau {int(percentis[0])+1}-{int(percentis[1])}",
                             f"Grau {int(percentis[1])+1}-{int(percentis[2])}", f"Grau {int(percentis[2])+1}-{int(percentis[3])}",
                             f"Grau > {int(percentis[3])}"]
            
            cmap = plt.cm.plasma
            legend_colors = [cmap(0.1), cmap(0.3), cmap(0.5), cmap(0.7), cmap(0.9)]

            legend_elements = [Patch(facecolor=legend_colors[i], label=legend_labels[i], alpha=0.8) for i in range(5)]

            plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1.0), title="Legenda - Cores por Grau",
                  fontsize=10, framealpha=0.9)

            plt.title("REDE FACEBOOK", fontsize=16, pad=20)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            return True
            
        except Exception as e:
            print(f"Erro na visualização: {e}")
            return False
        
    def visualizar_comunidades(self):
        try:
            if self.G_subset is None:
                print("O subgrafo não existe!")
                return False

            pos = nx.spring_layout(self.G_subset, k=2, iterations=1000)

            #Calcula o grau de cada nó coloca em um dicionário e o tamanho do nó é proporcional ao grau
            graus = dict(self.G_subset.degree())
            tamanhos = [np.log(graus[node]+1)*25 for node in self.G_subset.nodes()]

            cmap = plt.get_cmap("plasma", max(self.communities.values())+1)

            #A cor de cada nó é a partir da lista de comunidades
            nx.draw_networkx_nodes(self.G_subset, pos, node_color=list(self.communities.values()), cmap=cmap, node_size=tamanhos, alpha=1)
            nx.draw_networkx_edges(self.G_subset, pos, alpha=1, edge_color="gray", width=0.5)

            plt.axis("off")
            plt.title("COMUNIDADES")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Erro na visualização: {e}")
            return False

def main():
    graph = FacebookGraph()

    if not graph.baixar_dados():
        return
    
    if not graph.carregar_rede():
        return
    
    if not graph.extrair_subconjunto(2000):
        return
    
    if not graph.calcular_metricas():
        return

    if not graph.visualizar_rede():
        return
    
    if not graph.visualizar_comunidades():
        return

if __name__ == "__main__":
    main()
