import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from load_dataset import load_graph, load_dataset
import graph_reduction as gred
# Test
test = load_graph.Beam2D()
GFDS = load_dataset.Beam2D()
# test = load_graph.Beam3D()
# GFDS = load_dataset.Beam3D()
# test = load_graph.Fibonacci()
# GFDS = load_dataset.Fibonacci()
# test = load_graph.Plane()
# GFDS = load_dataset.Plane()
G = test.Gcoor
original_graph = G
y = GFDS.y
weights_last = y[-1]
weights_apply = weights_last / weights_last.max()
for _edge_ in G.edges:  G[_edge_[0]][_edge_[1]]['weight'] = np.abs(weights_apply[_edge_[0] - 1])/2
# for edge in G.edges(): G[edge[0]][edge[1]]['weight'] = np.random.randint(1, 10)
L = nx.laplacian_matrix(G, weight='weight').toarray()
# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(L)
# k = 10 # Select a subset (for example, the first k eigenvalues)
# k = int(len(eigenvalues)*0.95) # Select a subset (for example, the first k eigenvalues)
threshold = .1 # dam to jako gamma parameter zas ? 
# name_experiment = 'WL1'
# k = int(len(eigenvalues)*0.8) # Select a subset (for example, the first k eigenvalues)
name_experiment = 'WL2'
k = int(len(eigenvalues)*0.6) # Select a subset (for example, the first k eigenvalues)1 wl2


selected_eigenvalues = eigenvalues[:k]
selected_eigenvectors = eigenvectors[:, :k]
reduced_graph = nx.Graph() # Form a reduced graph using the selected eigenvectors
reduced_graph.add_nodes_from(range(1, len(G.nodes) + 1)) # Add nodes to the reduced graph
positions = {i + 1: (selected_eigenvectors[i, 0], selected_eigenvectors[i, 1]) for i in range(len(G.nodes))} 
# threshold = .1 # dam to jako gamma parameter zas ? 


top_indices = np.argsort(eigenvalues)[-k:]
edges_to_keep = [(i, j) for i, j, data in original_graph.edges(data=True) if data.get('weight', 1) > threshold]
reduced_graph.add_nodes_from(positions)
reduced_graph.add_edges_from(edges_to_keep)
a_0    = nx.adjacency_matrix(G).toarray()
a_red  = nx.adjacency_matrix(reduced_graph).toarray()
graph = reduced_graph
ltm = np.tril(np.random.rand(a_0.shape[0], a_0.shape[1]))
s = ltm * a_red.T 
I =  np.ones([1, a_0.shape[0]])
degree_array = np.array(list(reduced_graph.degree))
degree_array = degree_array[:,1] * I
a = degree_array
e0 = np.where(a < 1 )[1] + 1
e1 = np.where(a < 2 )[1] + 1
e_ = np.concatenate([e0,e1])
eu = np.unique(e_)
ea = list(np.array([eu,np.roll(eu,1)]).T)
reduced_graph.add_edges_from(ea)
# %%
from pathlib import Path
# print(f'fully conneccted: {nx.is_connected(reduced_graph)}')
# gred.plotter(G, reduced_graph)
reduced_graph.name = f'{test.path_.stem}_{name_experiment}'
# reduced_graph.name = f'{test.path_.stem}_WL2'
path_adjlist = f'{test.path_.parent}/{reduced_graph.name}.adjlist'
if Path(path_adjlist).is_file() == False:
    nx.write_adjlist(reduced_graph, path_adjlist)
