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
G =          preprocessing.create_graph(d_nodes, d_elements) # create graph connections basic biderictional
#%% how to add coordinates for graph
# Gc = preprocessing.create_graph_coor_2D(d_nodes, d_elements)
# Gc = preprocessing.create_graph_coor_tetra(d_nodes, d_elements)
Gc = preprocessing.create_graph_coor(d_nodes, d_elements) # C3D8 Hexa
#%% create position ... 
pos= d_nodes[['x','y', 'z']].values 
pos=nx.get_node_attributes(Gc,'pos')
#%% 
from load_dataset import load_dataset 
GFDS = load_dataset.Beam3D()
gfds_name   = GFDS.gfds_name
pathRes     = GFDS.pathRes
D           = GFDS.D

X0 = GFDS.X0
y = GFDS.y
# %%
weights_last = y[-1]
i = 0
for _edge_ in G.edges:
    
    _source, _target = _edge_[0], _edge_[1]
    
    # wl_ = weights_last[_target - 1] - weights_last[_source -1]
    wl_ = weights_last[_source- 1]
    # G[_source][_target]['weight'] = weights_last[i]
    G[_source][_target]['weight'] = np.abs(wl_)/2
    
    print(f"{_edge_}{wl_}")
    # i+=1
# for _edge_ in G.edges:
    # print(_edge_['weight'])
    # _source, _target = _edge_[0], _edge_[1]
    # G[_source][_target]['weight'] 
    # %%
p1 = nx.shortest_path(G, source=1, target=25, ) # b2
p2 = nx.shortest_path(G, source=1, target=25, method='dijkstra')
p3 = nx.shortest_path(G, source=1, target=1, weight='weight')
# %%
weights_last = y[-1]
i = 0
for _edge_ in Gc.edges:
    _source, _target = _edge_[0], _edge_[1]
    wl_ = weights_last[_target - 1 ] - weights_last[_source - 1]
    Gc[_source][_target]['weight'] = np.abs(wl_)/2
    print(f"{_edge_}{wl_}")


