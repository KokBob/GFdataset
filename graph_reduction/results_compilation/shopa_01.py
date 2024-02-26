# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:12:24 2024
@author: marek
na zamysleni 
https://stackoverflow.com/questions/50723854/networkx-finding-the-shortest-path-to-one-of-multiple-nodes-in-graph
"""
import sys
sys.path.append(".") #
# sys.path.append("..") #esik
import networkx as nx
from load_dataset import load_dataset 
from gnsfem import preprocessing
import graph_reduction as gred
# import graph_preprocessing as grep
# from gnsfem import visualization
#%% select tes
file_name = r'../datasets/b2/Beam2D.inp'  # 2D mesh
file1       = open(file_name,"r")
d           = file1.readlines() 
#%% get lines number
endLiner = '*Nset, nset=Set-Part, generate'
line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = endLiner)# b2
d_nodes         = preprocessing.get_nodes(file_name, line_elements,line_nodes)
d_elements      = preprocessing.get_edges(file_name, line_elements,line_nodes, line_end)
Gc              = preprocessing.create_graph_coor_2D(d_nodes, d_elements)
#%% create position ... 
pos= nx.get_node_attributes(Gc,'pos')
GFDS        = load_dataset.Beam2D()
Gc = gred.get_graph_weighted(graph_gfem=Gc, weights_apply=GFDS.y[-1])
pos= d_nodes[['x','y']].values #b2
pos= nx.get_node_attributes(Gc,'pos')
# %% deving
g = gred.gra(Gc)
psp  = g.get_shortest_path()
pts  = g.get_travel_salesman(Gc)
gsh =g.get_graph_reduced(path_for_reduction=psp)
gts =g.get_graph_reduced(path_for_reduction=pts)
# gred.plotter(Gc,gsh, pos = pos)
# %%
gred.plotter(Gc,gts, pos = pos)
# p = g.get_shortest_path()
# gd = g.G_dummy
# get_info(g.G_dummy)
# get_info(g.G_red)
# %%
gr1 = gred.gra(Gc)
gr2 = gred.gra(Gc)
# p1 = gr1.get_shortest_path()
# p2 = gr2.get_shortest_path()
# %%
# nx.draw_networkx(gsh)
# g2r = gr2.get_travel_salesman(graph_weighted=Gc)
# gred.plotter(Gc,g1r)
# gred.plotter(Gc,g2r)
# nx.draw_networkx(Gc, pos=pos,with_labels=True, font_weight='bold', node_size=700, font_size=10)
