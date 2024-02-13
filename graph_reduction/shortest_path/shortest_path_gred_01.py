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
file_name = r'../datasets/fs/Fibonacci.inp'  # Fibonacci
file_name = r'../datasets/pl/Plane.inp'  # Fibonacci
file1       = open(file_name,"r")
d           = file1.readlines() 
#%% get lines number

# line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = '*Nset, nset=Set-Part, generate')
# line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = '*Elset, elset=Set-All, generate') # b3,Fs
line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = '*Nset, nset=RC-1_Set-All-Material, generate')
# %%
# line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, ) # b3 je tu bug nejakej 
d_nodes         = preprocessing.get_nodes(file_name, line_elements,line_nodes)
d_elements      = preprocessing.get_edges(file_name, line_elements,line_nodes, line_end)
# %% tady je bug pro b3
G =          preprocessing.create_graph(d_nodes, d_elements) # create graph connections basic biderictional
#%% how to add coordinates for graph
# Gc = preprocessing.create_graph_coor_2D(d_nodes, d_elements)
Gc = preprocessing.create_graph_coor_tetra(d_nodes, d_elements)
# Gc = preprocessing.create_graph_coor(d_nodes, d_elements) # C3D8 Hexa
#%% create position ... 
pos= d_nodes[['x','y', 'z']].values 
pos=nx.get_node_attributes(Gc,'pos')
#%% 
# nx.draw_networkx(Gc)
# nx.draw_networkx(G)
# nx.draw_networkx(Gc, pos=pos)
# %% 2D
# shortest path unweighetd and wirghted 
# difference sbetween algs 
# exclude bc nodes and loading nodes algos 
# ... fully connected as in benchmark ... `
# p1 = nx.shortest_path(Gc, source=1, target=33, ) # b2
# p2 = nx.shortest_path(Gc, source=1, target=33, method='dijkstra')
# p2 = nx.shortest_path(G, source=1, target=33, method='dijkstra')
# p3 = nx.shortest_path(Gc, source=1, target=33, method='bellman-ford') # pres Gc kde je cesta fakt vezme nejkratsi a ne pres vsechny nody,
# p3 = nx.shortest_path(G, source=1, target=33, method='bellman-ford') # b2
# %% b3
p3 = nx.shortest_path(G, source=1, target=(len(G.nodes)-1), method='dijkstra') #fs
# p3 = nx.shortest_path(G, source=1, target=1426, method='dijkstra') #fs
# p3 = nx.shortest_path(G, source=1, target=25, method='dijkstra') #b3
# asi to fakt konci na poslednim nodu a je mu uplne jedno jaky je cilovy ..
# musel bych to nejal prevzorkovat 
# p3 = nx.shortest_path(G, source=1, target=13, method='bellman-ford') # b3
 # If weight is None, unweighted graph methods are used, and this suggestion is ignored.
 # jak to tam zakomponovat ... 
 # %%
# benchmark 
Gd = nx.Graph()
for nd in d_nodes.index:    
    src0 = d_nodes.iloc[nd]
    src = int(src0['#Node'])
    pos0 = (src0['x'], src0['y'], src0['z'])
    Gd.add_node(src, pos = pos0)
# for i in range(np.shape(d_elements.index)[0]):
#     els0 = d_elements.iloc[i].values 
#     els1 = np.roll(els0,1)
#     elsf = np.array([els0,els1]).T
#     Gd.add_edges_from(elsf)
# return Gd
# %%
#opravdu nejkratsi cesta 
# a2 = np.array([p1,np.roll(p1,1)])
# a2 = np.array([p2,np.roll(p2,1)])
a2 = np.array([p3,np.roll(p3,1)])
a2 = a2[:,1:]
a3 = a2.T.tolist()
Gd.add_edges_from(a3)
# Gd.add_nodes_from([p1])
# nx.draw_networkx(Gd)
# nx.draw_networkx(Gd, pos=pos)

# %%
# Gd.name = './datasets/b2/Beam2D_ShP'
# Gd.name = '../datasets/b3/Beam3D_ShP'
# Gd.name = '../datasets/fs/Fibonacci_ShP'
Gd.name = '../datasets/pl/Plane_ShP'
nx.write_adjlist(Gd, Gd.name + ".adjlist")
