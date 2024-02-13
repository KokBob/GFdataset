import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from load_dataset import load_graph
import graph_reduction as gred
# Test
test = load_graph.Beam2D()
# test = load_graph.Beam3D()
# test = load_graph.Fibonacci()
# test = load_graph.Plane()
G = test.Gcoor
# G = test.G0
original_graph = G
for edge in G.edges(): G[edge[0]][edge[1]]['weight'] = np.random.randint(1, 10)
L = nx.laplacian_matrix(G, weight='weight').toarray()
# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(L)
# k = 10 # Select a subset (for example, the first k eigenvalues)
k = int(len(eigenvalues)*0.8) # Select a subset (for example, the first k eigenvalues)
selected_eigenvalues = eigenvalues[:k]
selected_eigenvectors = eigenvectors[:, :k]
reduced_graph = nx.Graph() # Form a reduced graph using the selected eigenvectors
reduced_graph.add_nodes_from(range(1, len(G.nodes) + 1)) # Add nodes to the reduced graph
positions = {i + 1: (selected_eigenvectors[i, 0], selected_eigenvectors[i, 1]) for i in range(len(G.nodes))} # Add edges based on the positions determined by the eigenvectors
# reduced_graph.add_edges_from(G.edges())
# %%
# threshold = 1 
# threshold = 5 # dam to jako gamma parameter zas ? 
# threshold = 6 # dam to jako gamma parameter zas ? 
threshold = 5 # dam to jako gamma parameter zas ? 
# threshold = 2 # dam to jako gamma parameter zas ? 
top_indices = np.argsort(eigenvalues)[-k:]
edges_to_keep = [(i, j) for i, j, data in original_graph.edges(data=True) if data.get('weight', 1) > threshold]
# is in the graph connection ? 
# I all connected nodes 
# ho to test it ? kde chybi edge ?
# RL to catch right value ? is this deterministic 
reduced_graph.add_nodes_from(positions)
reduced_graph.add_edges_from(edges_to_keep)
# %%
# Je fully connected test ? how to find if graph is fully connected (FC) ...
# ... FC: test travel salesman
# test_fc_ts = nx.he(reduced_graph)
# Plot the original and reduced graphs
a_0    = nx.adjacency_matrix(G).toarray()
a_red  = nx.adjacency_matrix(reduced_graph).toarray()
# %%
graph = reduced_graph
# powers = [np.linalg.matrix_power(a_red, i) for i in range(1, len(graph.nodes))]
# Sum the powers
# so( nx.adjacency_matrix(graph).toarray())
# sum_powers = sum(powers)
# lower_triangular_matrix = np.tril(np.random.rand(a_0.shape[0], a_0.shape[1]))
ltm = np.tril(np.random.rand(a_0.shape[0], a_0.shape[1]))
s = ltm * a_red.T 
I =  np.ones([1, a_0.shape[0]])
# nds0# timhle bych mel dostat ktere nody jsou nepropojene 
# nds0= sum(np.linalg.matrix_power(s,1)).reshape(1,a_0.shape[0])

# powers = [np.linalg.matrix_power(a_red, i) for i in range(1, len(graph.nodes))]
# !!!  we are suggesting convolutional sparsification 
# porovnani 

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
p1 = Path(test.input_file_name)
print(f'fully conneccted: {nx.is_connected(reduced_graph)}')
gred.plotter(G, reduced_graph)
reduced_graph.name = f'{p1.stem}_L1'
path_adjlist = f'{p1.parent}/{reduced_graph.name}.adjlist'
nx.write_adjlist(reduced_graph, path_adjlist)
