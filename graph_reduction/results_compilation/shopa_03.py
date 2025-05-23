# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:12:24 2024
@author: marek
na zamysleni 
https://stackoverflow.com/questions/50723854/networkx-finding-the-shortest-path-to-one-of-multiple-nodes-in-graph
"""
import sys
sys.path.append(".") #
import networkx as nx
import graph_reduction as gred
from load_dataset import load_graph, load_dataset
groot = '../datasets/b2/Beam2D'
G0 = load_dataset.Beam2D()
G1 = load_dataset.Beam2D(path_graph = f'{groot}_ShP.adjlist')
G2 = load_dataset.Beam2D(path_graph = f'{groot}_TrS.adjlist')
G3 = load_dataset.Beam2D(path_graph = f'{groot}_L1.adjlist')
G4 = load_dataset.Beam2D(path_graph = f'{groot}_WL2.adjlist')

# %%
# G0 = GS.G
# Gc = GS.g_
# Gc = GS.Gcoor
gred.plotter5(G0.G,G1.G,G2.G,G3.G,G4.G)
# gred.get_info(GFDS.g_)
