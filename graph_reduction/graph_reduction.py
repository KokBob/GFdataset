
# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
def get_info(graph_):
    print(f'Nodes: {graph_.nodes.data()} \n')
    print(f'Edges: {graph_.edges.data()} \n')
def get_graph_weighted(graph_gfem, weights_apply):
    for _edge_ in graph_gfem.edges: 
        graph_gfem[_edge_[0]][_edge_[1]]['weight'] = np.abs(weights_apply[_edge_[0] - 1])/2
    return graph_gfem
def plotter(graph_original, graph_reduced, pos = None):
    '''
    chybi tu pozice pos

    Parameters
    ----------
    graph_original : TYPE
        DESCRIPTION.
    graph_reduced : TYPE
        DESCRIPTION.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    node_size=400
    s1 = str(len(graph_original.edges.data()))
    s2 = str(len(graph_reduced.edges.data()))
    # Plot the original and reduced graphs
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    nx.draw_networkx(graph_original, pos = pos,with_labels=True, font_weight='bold', 
                     node_size=node_size, font_size=10)
    plt.title('Original Fully Connected Graph')
    plt.text(x = 1, y = 1, s = s1 )
    # plt.text()
    
    plt.subplot(1, 2, 2)
    nx.draw_networkx(graph_reduced, pos=pos, with_labels=True, font_weight='bold', 
                     node_size=node_size, font_size=10)
    plt.text(x = 1, y = 1, s = s2 )
    # plt.title(f'Reduced Graph (Laplacian, k={k})')
    # plt.title(f'Reduced Graph ()')
    print(f'fully conneccted: {nx.is_connected(graph_reduced)}')
    plt.show()   
def plotter3(graph_original, 
             graph_reduced, 
             graph_reduced_2, 
             pos = None):
    '''
    chybi tu pozice pos

    Parameters
    ----------
    graph_original : TYPE
        DESCRIPTION.
    graph_reduced : TYPE
        DESCRIPTION.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    node_size=400
    s1 = str(len(graph_original.edges.data()))
    s2 = str(len(graph_reduced.edges.data()))
    s3 = str(len(graph_reduced_2.edges.data()))
    # Plot the original and reduced graphs
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    nx.draw_networkx(graph_original, pos = pos,with_labels=True, font_weight='bold', 
                     node_size=node_size, font_size=10)
    plt.title('Original Fully Connected Graph')
    plt.text(x = .01, y = .01, s = s1 )
    # plt.text()
    
    plt.subplot(1, 3, 2)
    nx.draw_networkx(graph_reduced, pos=pos, with_labels=True, font_weight='bold', 
                     node_size=node_size, font_size=10)
    plt.title(f'Graph {graph_reduced.name}')
    plt.text(x = .01, y = .01, s = s2 )
    
    plt.subplot(1, 3, 3)
    nx.draw_networkx(graph_reduced_2, pos=pos, with_labels=True, font_weight='bold', 
                     node_size=node_size, font_size=10)
    plt.title(f'Graph {graph_reduced_2.name}')
    plt.text(x = .01, y = .01, s = s3 )
    # plt.title(f'Reduced Graph (Laplacian, k={k})')
    # plt.title(f'Reduced Graph ()')
    
    plt.show()   
class gra:
    def __init__(self, G_orig):
        self.G = G_orig
        self.G_temp = nx.Graph()
        self.G_temp.add_nodes_from(self.G)
        # self.G_dummy = self.G_red
        # self.G_dummy.add_edges_from(self.G.edges)

        # pass
    def add_weights(self, weights_apply):
        for _edge_ in self.G.edges: 
            self.G[_edge_[0]][_edge_[1]]['weight'] = np.abs(weights_apply[_edge_[0] - 1])/2
        pass
    def get_graph_reduced(self, path_for_reduction):
        # G_red = self.G_temp   
        G_red = nx.Graph()
        G_red.add_nodes_from(self.G)
        a2 = np.array([path_for_reduction,np.roll(path_for_reduction,1)])
        a3 = a2[:,1:].T.tolist()
        G_red.add_edges_from(a3)
        return G_red
    def get_shortest_path(self, ):
        G_dummy = self.G_temp
        G_dummy.add_edges_from(self.G.edges)

        # self.G_dummy.add_edges_from(self.G.edges)
        # self.path_shortest = nx.shortest_path(self.G, source=1,  
        # nese sebou data o pos, =>  nejkratsi ceta je pak ta geometricky nejkratsi
        # self.path_shortest = nx.shortest_path(self.G_dummy, 
        self.path_shortest = nx.shortest_path(G_dummy, 
                              source=1, 
                              target = list(G_dummy.nodes)[-1],)
                                              # weight = 'weight') 
        self.G_red_ShP = self.get_graph_reduced(self.path_shortest)
        # return self.G_red_ShP
        return self.path_shortest
    def get_travel_salesman(self, graph_weighted):
        # self.G_dummy.add_weighted_edges_from(graph_weighted)
        G_dummy = self.G_temp
        G_dummy.add_edges_from(self.G.edges)
        self.path_salesman = nx.approximation.traveling_salesman_problem(G_dummy, 
                weight='weight', nodes=None, cycle=True, method=None)
        self.G_red_TrS  = self.get_graph_reduced(self.path_salesman)
        return self.path_shortest
    def laplacik(self,):
        path_ = somepather(andItsPars)
        pass
    def get_info(self,):

        print(f'Nodes: {self.G_red.nodes.data()} \n')
        print(f'Edges: {self.G_red.edges.data()} \n')
