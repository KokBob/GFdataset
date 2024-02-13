import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def laplacian_reduction(original_graph, k=2, threshold=0.5):
    # Compute the Laplacian matrix with weights
    laplacian_matrix = nx.laplacian_matrix(original_graph, weight='weight').toarray()

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

    # Select a subset (for example, the first k eigenvalues)
    selected_eigenvectors = eigenvectors[:, -k:]

    # Compute the norm of each row in the selected eigenvectors
    row_norms = np.linalg.norm(selected_eigenvectors, axis=1)

    # Threshold for edge reduction
    edge_threshold = threshold * np.max(row_norms)

    # Determine edges to keep based on the threshold
    edges_to_keep = [(i, j) for i, j, data in original_graph.edges(data=True) if row_norms[i] > edge_threshold and row_norms[j] > edge_threshold]

    # Form a reduced graph with the selected edges
    reduced_graph = nx.Graph()

    # Add nodes to the reduced graph
    reduced_graph.add_nodes_from(range(1, len(original_graph.nodes) + 1))

    # Add edges to the reduced graph
    reduced_graph.add_edges_from([(i + 1, j + 1) for i, j in edges_to_keep])

    return reduced_graph

# Create a fully connected graph G with weighted edges
G = nx.complete_graph(5)
for i, j in G.edges():
    G[i][j]['weight'] = np.random.rand()

# Parameters for Laplacian reduction
k = 5
threshold = 0.95

# Reduce edges using Laplacian eigenmaps
reduced_graph = laplacian_reduction(G, k, threshold)



# Plot the original and reduced graphs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
nx.draw_networkx(G, with_labels=True, font_weight='bold', node_size=700, font_size=10)
plt.title('Original Fully Connected Graph')

plt.subplot(1, 2, 2)
nx.draw_networkx(reduced_graph, with_labels=True, font_weight='bold', node_size=700, font_size=10)
plt.title(f'Reduced Graph (Threshold={threshold})')

plt.show()
