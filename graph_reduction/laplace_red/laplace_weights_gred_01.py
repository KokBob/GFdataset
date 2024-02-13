# -*- coding: utf-8 -*-
"""
na zamysleni 
https://stackoverflow.com/questions/50723854/networkx-finding-the-shortest-path-to-one-of-multiple-nodes-in-graph
"""
import sys
sys.path.append(".") 
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gnsfem as gf
from gnsfem import preprocessing
# from gnsfem import visualization
#%% select tes
# file_name = r'../datasets/b2/Beam2D.inp'  # 2D mesh
file_name = r'../datasets/b3/Beam3D.inp'  # 3D mesh
# file_name = r'../datasets/fs/Fibonacci.inp'  # Fibonacci
# file_name = r'../datasets/pl/Plane.inp'  # Fibonacci
file1       = open(file_name,"r")
d           = file1.readlines() 
#%% get lines number
# line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = '*Nset, nset=Set-Part, generate')
line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = '*Elset, elset=Set-All, generate') # b3,Fs
# line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = '*Nset, nset=RC-1_Set-All-Material, generate')
# %%
# line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, ) # b3 je tu bug nejakej 
d_nodes         = preprocessing.get_nodes(file_name, line_elements,line_nodes)
d_elements      = preprocessing.get_edges(file_name, line_elements,line_nodes, line_end)
# %% tady je bug pro b3
G =     preprocessing.create_graph(d_nodes, d_elements) # create graph connections basic biderictional
# Gc =    preprocessing.create_graph_coor(d_nodes, d_elements) # C3D8 Hexa

# %%
# Compute the Laplacian matrix with weights
laplacian_matrix = nx.laplacian_matrix(G).toarray()

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

# Map nodes from the original graph to the reduced graph
node_mapping = {original_node: i + 1 for i, original_node in enumerate(G.nodes)}
# %%
# Check unique values in the eigenvectors
unique_values = set(np.concatenate(eigenvectors))
print("Unique values in eigenvectors:", unique_values)
# %%
# Check if all values in the eigenvectors have corresponding mappings
missing_keys = [value for value in unique_values if value not in node_mapping]
print("Missing keys in node_mapping:", missing_keys)
# %%
# Select a subset (for example, the first k eigenvalues)
k = 2
selected_eigenvalues = eigenvalues[:k]
selected_eigenvectors = eigenvectors[:, :k]
# %%
# Form a reduced graph using the selected eigenvectors
reduced_graph = nx.Graph()
# %%
reduced_graph.add_nodes_from(range(1, len(G.nodes) + 1))
# %%
for i, j in zip(selected_eigenvectors[:, 0], selected_eigenvectors[:, 1]):
    (node_mapping[i], node_mapping[j]
# %%
reduced_graph.add_edges_from([(node_mapping[i], node_mapping[j]) for i, j in zip(selected_eigenvectors[:, 0], selected_eigenvectors[:, 1])])
# %%
# Plot the original and reduced graphs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
nx.draw_networkx(G, with_labels=True, font_weight='bold', node_size=700, font_size=10)
plt.title('Original Graph')

plt.subplot(1, 2, 2)
nx.draw_networkx(reduced_graph, with_labels=True, font_weight='bold', node_size=700, font_size=10)
plt.title('Reduced Graph')

plt.show()
