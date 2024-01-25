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
# file_name = r'../datasets/b3/Beam3D.inp'  # 3D mesh
# file_name = r'../datasets/fs/Fibonacci.inp'  # Fibonacci
file_name = r'../datasets/pl/Plane.inp'  # Plane
file1       = open(file_name,"r")
d           = file1.readlines() 
#%% get lines number
# line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = '*Nset, nset=Set-Part, generate')# b2
# line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = '*Elset, elset=Set-All, generate') # b3,Fs
line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = '*Nset, nset=RC-1_Set-All-Material, generate')
# %%
d_nodes         = preprocessing.get_nodes(file_name, line_elements,line_nodes)
d_elements      = preprocessing.get_edges(file_name, line_elements,line_nodes, line_end)
# %% tady je bug pro b3
# Gc = preprocessing.create_graph_coor_2D(d_nodes, d_elements)
Gc = preprocessing.create_graph_coor_tetra(d_nodes, d_elements)
# Gc = preprocessing.create_graph_coor(d_nodes, d_elements) # C3D8 Hexa
#%% create position ... 
# pos= d_nodes[['x','y']].values #b2
pos= d_nodes[['x','y', 'z']].values 
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
for _edge_ in Gc.edges:
    _source, _target = _edge_[0], _edge_[1]
    # wl_ = weights_last[_target - 1] - weights_last[_source -1]
    wl_ = weights_last[_source- 1]
    # G[_source][_target]['weight'] = weights_last[i]`
    Gc[_source][_target]['weight'] = np.abs(wl_)/2
    print(f"{_edge_}{wl_}")

# https://stackoverflow.com/questions/63838078/plotting-networkx-graph-how-to-change-node-position-instead-of-resetting-every
# p4 = nx.shortest_path(Gc, source=1, weight = 'weight') #fs
# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.traveling_salesman.traveling_salesman_problem.html
p5 = nx.approximation.traveling_salesman_problem(Gc, weight='weight', nodes=None, cycle=True, method=None)
# p6 = nx.approximation.traveling_salesman_problem(Gc, weight='weight', nodes=None, cycle=False, method=None)


Gd = nx.Graph()
for nd in d_nodes.index:    
    src0 = d_nodes.iloc[nd]
    src = int(src0['#Node'])
    # pos0 = (src0['x'], src0['y'])
    pos0 = (src0['x'], src0['y'], src0['z'])
    Gd.add_node(src, pos = pos0)

# %%
#opravdu nejkratsi cesta 

a2 = np.array([p5,np.roll(p5,1)])
a2 = a2[:,1:]
a3 = a2.T.tolist()
Gd.add_edges_from(a3)

# %%
method_red = 'TrS'
# Gd.name = '../datasets/b2/Beam2D_ShP'
# Gd.name = '../datasets/b2/Beam2D_TrS'
# Gd.name = '../datasets/b3/Beam3D_TrS'
# Gd.name = '../datasets/b3/Beam3D_ShP'
# Gd.name = '../datasets/fs/Fibonacci_ShP'
# Gd.name = '../datasets/fs/Fibonacci_TrS'
Gd.name = '../datasets/pl/Plane_{method_red}'
nx.write_adjlist(Gd, Gd.name + ".adjlist")
