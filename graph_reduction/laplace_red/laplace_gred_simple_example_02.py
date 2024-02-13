import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from load_dataset import load_graph
import graph_reduction as gred
# Test
test = load_graph.Beam2D()
# G = test.Gcoor
G = test.G0
original_graph = G
for edge in G.edges(): G[edge[0]][edge[1]]['weight'] = np.random.randint(1, 10)
L = nx.laplacian_matrix(G, weight='weight').toarray()
# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(L)
k = 5 # Select a subset (for example, the first k eigenvalues)
selected_eigenvalues = eigenvalues[:k]
selected_eigenvectors = eigenvectors[:, :k]
# %%
# Form a reduced graph using the selected eigenvectors
reduced_graph = nx.Graph()
# Add nodes to the reduced graph
reduced_graph.add_nodes_from(range(1, len(G.nodes) + 1))
# Add edges based on the positions determined by the eigenvectors
positions = {i + 1: (selected_eigenvectors[i, 0], selected_eigenvectors[i, 1]) for i in range(len(G.nodes))}
# reduced_graph.add_edges_from(G.edges())

# %%
# threshold = 1 
# threshold = 5 # dam to jako gamma parameter zas ? 
threshold = 6 # dam to jako gamma parameter zas ? 
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
# %%
a_0    = nx.adjacency_matrix(G).toarray()
a_red  = nx.adjacency_matrix(reduced_graph).toarray()
# %%
def so(matrix_visualize):
    plt.figure()
    plt.imshow(matrix_visualize)
    # plt.title(f"{str(matrix_visualize)}")
    
so(a_0)
so(a_red)
# %%
graph = reduced_graph
powers = [np.linalg.matrix_power(a_red, i) for i in range(1, len(graph.nodes))]
# Sum the powers
so( nx.adjacency_matrix(graph).toarray())
sum_powers = sum(powers)
print(sum_powers)
gred.plotter(G, graph)
# %%
# lower_triangular_matrix = np.tril(np.random.rand(a_0.shape[0], a_0.shape[1]))
ltm = np.tril(np.random.rand(a_0.shape[0], a_0.shape[1]))
s = ltm * a_red.T 

# %%
so = plt.imshow
so(s)
# %%
# I =  np.ones([a_0.shape[0], 1])
I =  np.ones([1, a_0.shape[0]])
so(I)
# %%
# nds0# timhle bych mel dostat ktere nody jsou nepropojene 
nds0= sum(np.linalg.matrix_power(s,1)).reshape(1,33)
so(nds0)
# powers = [np.linalg.matrix_power(a_red, i) for i in range(1, len(graph.nodes))]
# %%
# !!!  we are suggesting convolutional sparsification 
# porovnani 
gred.plotter(G, reduced_graph)
leg = len(reduced_graph.edges)
print(leg)
# %%
import matplotlib.pyplot as plt
np.nonzero(nds0)
# %%
plt.imshow(s)
# plt.imshow(np.array([nds0,1]))
# %%

# %%
print(reduced_graph.degree)
# gred.plotter(G, reduced_graph)
degree_array = np.array(list(reduced_graph.degree))
degree_array = degree_array[:,1] * I
plt.imshow(degree_array)
# %%
nx.draw_networkx(reduced_graph)
# gred.plotter(G, reduced_graph)
# %%
a = degree_array
# np.where(a < 1 , a, -1) 
# np.where(a < 1 )[1] #to maximalne udelam ty co nejsou spojene
np.where(a == 1 )[1] #to maximalne udelam ty co nejsou spojene
gred.plotter(G, reduced_graph)
# %%
triangular_matrix =ltm 
plt.imshow(ltm)
# Get the size of the matrix
matrix_size = triangular_matrix.shape[0]

# Initialize an empty adjacency matrix
adjacency_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

# Set connections based on the non-zero elements in the triangular matrix
for i in range(matrix_size):
    for j in range(i + 1, matrix_size):
        if triangular_matrix[i, j] != 0:
            # Set the corresponding connection in the adjacency matrix
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1
plt.imshow(adjacency_matrix)
            # %%
# Example: Create a lower triangular matrix with random values
lower_triangular_matrix = np.tril(np.random.randint(0, 2, (33, 33)))

plt.imshow(lower_triangular_matrix)
# return adjacency_matrix

# %%
import numpy as np

def create_adjacency_from_triangular(triangular_matrix):
    # Get the size of the matrix
    matrix_size = triangular_matrix.shape[0]

    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # Set connections based on the non-zero elements in the triangular matrix
    for i in range(matrix_size):
        for j in range(i + 1, matrix_size):
            if triangular_matrix[i, j] != 0:
                # Set the corresponding connection in the adjacency matrix
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    return adjacency_matrix

# Create an adjacency matrix from the lower triangular matrix
adjacency_matrix = create_adjacency_from_triangular(lower_triangular_matrix)
plt.imshow(adjacency_matrix)

# %%
# basd = np.array([np.roll(np.where(a < 2 )[1]
plt.close()
degree_array = np.array(list(reduced_graph.degree))
degree_array = degree_array[:,1] * I
a = degree_array
e0 = np.where(a < 1 )[1] + 1
e1 = np.where(a < 2 )[1] + 1
nx.draw_networkx(reduced_graph)
e_ = np.concatenate([e0,e1])
eu = np.unique(e_)
# e0 = np.where(a < 2 )[1]
# v0 = np.array(np.roll(e0,1),e0)
ea = list(np.array([eu,np.roll(eu,1)]).T)
reduced_graph.add_edges_from(ea)
# degree_array = np.array(list(reduced_graph.degree))
# degree_array = degree_array[:,1] * I
# plt.imshow(degree_array)
# reduced_graph.add_edges_from(e0)
# nx.draw_networkx(reduced_graph)
# gred.plotter(G, reduced_graph)
# %%
# a = degree_array
# %%
# reduced_graph.add_edges_from(e0)
gred.plotter(G, reduced_graph)
