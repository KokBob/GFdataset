# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:12:24 2024
@author: marek
na zamysleni 
https://stackoverflow.com/questions/50723854/networkx-finding-the-shortest-path-to-one-of-multiple-nodes-in-graph
"""
import sys
# sys.path.append(".") 
sys.path.append("..") #esik
import networkx as nx
import pandas as pd
import numpy as np
from gnsfem import preprocessing
import graph_reduction as gred
# from gnsfem import visualization
#%% select tes
file_name = r'../datasets/b2/Beam2D.inp'  # 2D mesh
# file_name = r'../datasets/b3/Beam3D.inp'  # 3D mesh
# file_name = r'../datasets/fs/Fibonacci.inp'  # Fibonacci
# file_name = r'../datasets/pl/Plane.inp'  # Plane
file1       = open(file_name,"r")
d           = file1.readlines() 
#%% get lines number
# line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = '*Nset, nset=Set-Part, generate')# b2
endLiner = '*Nset, nset=Part-1-1_Set-Part, generate'
line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = endLiner)# b2
# line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = '*Elset, elset=Set-All, generate') # b3,Fs
# line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = '*Nset, nset=RC-1_Set-All-Material, generate')
# %%
d_nodes         = preprocessing.get_nodes(file_name, line_elements,line_nodes)
d_elements      = preprocessing.get_edges(file_name, line_elements,line_nodes, line_end)
# %% tady je bug pro b3
Gc = preprocessing.create_graph_coor_2D(d_nodes, d_elements)
# %%
G0 = nx.Graph()
G0.add_nodes_from(np.arange(1,34))
#opravdu nejkratsi cesta 
path_set = np.arange(1,34)
# path_set = p5
a2 = np.array([path_set,np.roll(path_set,1)])
a3 = a2[:,1:].T.tolist()
G0.add_edges_from(a3)
# G0.add_edges_from(Gc.edges)
# G0.add_edges_from(Gc.edges)

# Gc = preprocessing.create_graph_coor_tetra(d_nodes, d_elements)
# Gc = preprocessing.create_graph_coor(d_nodes, d_elements) # C3D8 Hexa
#%% create position ... 
pos= d_nodes[['x','y']].values #b2
# pos= d_nodes[['x','y', 'z']].values 
pos=nx.get_node_attributes(Gc,'pos')
#%% 
from load_dataset import load_dataset 
# GFDS = load_dataset.Beam2D()
# GFDS = load_dataset.Beam3D()
# GFDS = load_dataset.Fibonacci()
GFDS = load_dataset.Plane()
gfds_name   = GFDS.gfds_name
pathRes     = GFDS.pathRes
D           = GFDS.D
X0 = GFDS.X0
y = GFDS.y

weights_last = y[-1]
# %%
# nx.draw_networkx(Gc)
# %%
wl = 0
for _edge_ in G0.edges:
    _source, _target = _edge_[0], _edge_[1]
    # wl_ = weights_last[_target - 1] - weights_last[_source -1]
    # wl_ = weights_last[_source- 1]
    # G[_source][_target]['weight'] = weights_last[i]`
    G0[_source][_target]['weight'] = wl*0.1
    wl +=1
    
for _edge_ in Gc.edges:
    _source, _target = _edge_[0], _edge_[1]
    wl_ = weights_last[_source- 1]
    Gc[_source][_target]['weight'] = np.abs(wl_)/2
    # print(f"{_edge_}{wl_}")

# https://stackoverflow.com/questions/63838078/plotting-networkx-graph-how-to-change-node-position-instead-of-resetting-every
# p4 = nx.shortest_path(Gc, source=1, target =len(Gc.nodes), weight = 'weight') #fs
# p4 = nx.shortest_path(G0, source=1, target =33, weight = 'weight') #fs

# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.traveling_salesman.traveling_salesman_problem.html
p5 = nx.approximation.traveling_salesman_problem(Gc, weight='weight', nodes=None, cycle=True, method=None)
# p6 = nx.approximation.traveling_salesman_problem(Gc, weight='weight', nodes=None, cycle=False, method=None)

# %%
G_red = nx.Graph()
G_red.add_nodes_from(np.arange(1,33))
path_set = p5
a2 = np.array([path_set,np.roll(path_set,1)])
a3 = a2[:,1:].T.tolist()
G_red.add_edges_from(a3)
# %%
# gred.plotter(Gc,G_red)
# gred.plotter(Gc,G0)
# gred.plotter(Gc,G_red)
G_red.name = 'TrS'
G0.name = 'ShP'
gred.plotter3(Gc,G0, G_red)

# gred.plotter(Gc,G0,pos)
# gred.plotter(Gc,G_red, pos)
# %%
# d = np.array(Gc.degree)
a = nx.adjacency_matrix(Gc, weight='weight').toarray()
l = nx.laplacian_matrix(Gc, weight='weight').toarray()
d = nx.degree(Gc)
# %%
d_ = l-a
# %%
# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(l)
# %%
k = 10
top_indices = np.argsort(eigenvalues)[-k:]
# %%
# Determine edges to keep based on the top indices
edges_to_keep = [(i, j) for i, j, data in Gc.edges(data=True) if i in top_indices and j in top_indices]

# %%
# method_red = 'TrS'
method_red = 'TrS'
Gd = G_red
# Gd.name = '../datasets/b2/Beam2D_ShP'
# Gd.name = '../datasets/b2/Beam2D_TrS'
# Gd.name = '../datasets/b3/Beam3D_TrS'
# Gd.name = '../datasets/b3/Beam3D_ShP'
# Gd.name = '../datasets/fs/Fibonacci_ShP'
# Gd.name = '../datasets/fs/Fibonacci_TrS'
# Gd.name = '../datasets/pl/Plane_{method_red}'
# nx.write_adjlist(Gd, Gd.name + ".adjlist")

