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
# 
# %%
    
# G = nx.Graph([(1, 2), (2, 3), (3, 4)])

# weight = 0.1

# Grow graph by one new node, adding edges to all existing nodes.
# wrong way - will raise RuntimeError
# G.add_weighted_edges_from(((5, n, weight) for n in G.nodes))
# correct way - note that there will be no self-edge for node 5

# G.add_weighted_edges_from(list((5, n, weight) for n in G.nodes))

# G.add_weighted_edges_from(list((5, n, weight) for n in G.nodes))


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
    # i+=1
    # break
# %%
for _edge_ in Gc.edges:
    # pass
    print(_edge_)
    # _source, _target = _edge_[0], _edge_[1]
    # G[_source][_target]['weight'] 
    
    
# %%

pc1 = nx.shortest_path(Gc, source=1, target=24, ) # b2
pc2 = nx.shortest_path(Gc, source=1, target=24, method='dijkstra')
pc3 = nx.shortest_path(Gc, source=1, target=24, weight='weight')
# weight2
# %%
nx.set_edge_attributes(G, values = weights_last, name = 'weight')
# %%
# nx.set_edge_attributes(G, values = weights, name = 'weight')
# nx.draw_networkx(Gc)
nx.draw_networkx(G)
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
p3 = nx.shortest_path(G, source=1, target=(len(G.nodes)-1), weight = 'weights') #fs
# p3 = nx.shortest_path(G, source=1, target=(len(G.nodes)-1), weight = weights_last) #fs
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
# Gd.name = '../datasets/pl/Plane_ShP'
# nx.write_adjlist(Gd, Gd.name + ".adjlist")
# %%
Ge = nx.Graph()
nodes_test = d_nodes.iloc[:5,:]
# %%
for nd in nodes_test.index:    
    src = nodes_test.iloc[nd,:]
    src_label = src['#Node']
    # src_pos = (src['x'], src['y'], src['z'])
    Ge.add_node(src_label) # zatim bez pozice 
    # Ge.add_node(src, pos = pos0)
nx.draw_networkx(Ge,)
# je potreba dodelat hrany
# nx.draw_networkx(Ge, pos=pos)
# nx.get_edge_attributes(Ge,'weight')
# %%    
weights_last = y[-1]
i = 0
# nodes_test
# nejsou tu zadne hrany, proto to nejde
# for _edge_ in Ge.edges:    
    # _source, _target = _edge_[0], _edge_[1]
    # wl_ = weights_last[_target - 1 ] - weights_last[_source - 1]
    # Ge[_source][_target]['weight'] = np.abs(wl_)/2
    # Ge.add_edge(_source, _target, weight=wl_)
    # print(f"{_edge_}{wl_}")


nx.draw_networkx(Ge,)
# %%


# G[0[object_id]['weight'] += 1

nx.draw_networkx(Gc)
# %%
for _edge_ in Gc.edges:
    
    _source, _target = _edge_[0], _edge_[1]
    
    # wl_ = weights_last[_target - 1] - weights_last[_source -1]
    wl_ = weights_last[_source- 1]
    # G[_source][_target]['weight'] = weights_last[i]
    Gc[_source][_target]['weight'] = np.abs(wl_)/2
    
    print(f"{_edge_}{wl_}")
nx.draw_networkx(Gc)
# %%
labels_weights = nx.get_edge_attributes(Gc,'weight')
pos_r=nx.fruchterman_reingold_layout(Gc)
nx.draw_networkx(Gc,pos = pos_r,with_labels=True)
nx.draw_networkx_edge_labels(Gc,pos = pos_r,edge_labels=labels_weights)
# %%
# podminka , musi byt fully connected co se tyce propojeni nodu 
# zadna redukce uzlu 
# p4 = nx.shortest_path(Gc, source=1, target=(len(Gc.nodes)-1), weight = 'weight') #fs
# https://stackoverflow.com/questions/63838078/plotting-networkx-graph-how-to-change-node-position-instead-of-resetting-every
p4 = nx.shortest_path(Gc, source=1, weight = 'weight') #fs
# for pos_ in pos: 
    # print(pos[pos_[:-1]])
# %%
# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.traveling_salesman.traveling_salesman_problem.html
p5 = nx.approximation.traveling_salesman_problem(Gc, weight='weight', nodes=None, cycle=True, method=None)

