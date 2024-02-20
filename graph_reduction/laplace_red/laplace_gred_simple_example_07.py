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
y = GFDS.y
weights_apply = y[-1] / y[-1].max()
reduced_graph = gred.spectral_reduction(G, weights_apply)
# %%
# from pathlib import Path
# print(f'fully conneccted: {nx.is_connected(reduced_graph)}')
# print(f'reduction: {len(reduced_graph.edges())/len(G.edges())}')
gred.plotter(G, reduced_graph)
# %%
# reduced_graph.name = f'{test.path_.stem}_{name_experiment}'
# reduced_graph.name = f'{test.path_.stem}_WL2'
# path_adjlist = f'{test.path_.parent}/{reduced_graph.name}.adjlist'
# if Path(path_adjlist).is_file() == False:
# nx.write_adjlist(reduced_graph, path_adjlist)
