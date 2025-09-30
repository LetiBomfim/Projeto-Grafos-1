import networkx as nx
import requests
import numpy as np
import matplotlib.pyplot as plt


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
            
            print("Arquivo aberto com sucesso!")
            
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

        print(f" - Nós: {self.G_subset.number_of_nodes()}")
        print(f" - Arestas: {self.G_subset.number_of_edges()}")
        print(f" - Densidade: {nx.density(self.G_subset):.6f}")

        return True
    
    def visualizar_rede(self):
        try:
            if self.G_subset is None:
                print("O subgrafo não existe!")
                return False
            
            plt.figure(figsize=(20, 15))

            #O iterations tá como 1000, porque não sei se meu computador aguenta
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

            plt.title("REDE FACEBOOK", fontsize=16, pad=20)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            return True
            
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
    
    if not graph.visualizar_rede():
        return

if __name__ == "__main__":
    main()