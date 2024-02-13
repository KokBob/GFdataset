import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Create a graph G with weighted edges
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4, 5])

# Add weighted edges using the 'weight' attribute
G.add_edge(1, 2, weight=2)
G.add_edge(2, 3, weight=1)
G.add_edge(3, 4, weight=3)
G.add_edge(4, 5, weight=2)
G.add_edge(1, 3, weight=4)
G.add_edge(2, 4, weight=1)

# Compute the Laplacian matrix with weights
laplacian_matrix = nx.laplacian_matrix(G, weight='weight').toarray()

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

# Select a subset (for example, the first k eigenvalues)
k = 2
selected_eigenvalues = eigenvalues[:k]
selected_eigenvectors = eigenvectors[:, :k]

# Form a reduced graph using the selected eigenvectors
reduced_graph = nx.Graph()

# Add nodes to the reduced graph
reduced_graph.add_nodes_from(range(1, len(G.nodes) + 1))

# Add edges based on the positions determined by the eigenvectors
positions = {i + 1: (selected_eigenvectors[i, 0], selected_eigenvectors[i, 1]) for i in range(len(G.nodes))}
reduced_graph.add_edges_from(G.edges())
reduced_graph.add_nodes_from(positions)



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
