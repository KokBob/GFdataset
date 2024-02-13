import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def laplacian_reduction(original_graph, k=2):
    # Compute the Laplacian matrix with weights
    laplacian_matrix = nx.laplacian_matrix(original_graph, weight='weight').toarray()

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

    # Select a subset (for example, the first k eigenvalues)
    selected_eigenvectors = eigenvectors[:, :k]

    # Form a reduced graph using the selected eigenvectors
    reduced_graph = nx.Graph()

    # Add nodes to the reduced graph
    reduced_graph.add_nodes_from(range(1, len(original_graph.nodes) + 1))

    # Add edges based on the positions determined by the eigenvectors
    positions = {i + 1: (selected_eigenvectors[i, 0], selected_eigenvectors[i, 1]) for i in range(len(original_graph.nodes))}
    reduced_graph.add_edges_from(original_graph.edges())
    reduced_graph.add_nodes_from(positions)

    return original_graph, reduced_graph, positions

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
G.add_edge(2, 1, weight=2)

# Laplacian reduction
original_graph, reduced_graph, positions = laplacian_reduction(G)

# Count of edges
num_edges_original = len(original_graph.edges())
num_edges_reduced = len(reduced_graph.edges())

# Plot the original and reduced graphs
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
nx.draw_networkx(original_graph, with_labels=True, font_weight='bold', node_size=700, font_size=10)
plt.title(f'Original Graph\nEdges: {num_edges_original}')

plt.subplot(1, 3, 2)
nx.draw_networkx(reduced_graph, with_labels=True, font_weight='bold', node_size=700, font_size=10, pos=positions)
plt.title(f'Reduced Graph\nEdges: {num_edges_reduced}')

plt.subplot(1, 3, 3)
plt.bar(['Original Graph', 'Reduced Graph'], [num_edges_original, num_edges_reduced], color=['blue', 'orange'])
plt.ylabel('Number of Edges')
plt.title('Edge Count Comparison')

plt.show()
