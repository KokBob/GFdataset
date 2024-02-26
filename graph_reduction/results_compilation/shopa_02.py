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
from load_dataset import load_graph
GS = load_graph.Beam2D()
G0 = GS.G0
Gc = GS.Gcoor
gred.plotter(G0,Gc)
gred.get_info(GS.g_)
