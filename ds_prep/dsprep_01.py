# -*- coding: utf-8 -*-
"""

load X .. CF and RF 
load G 
add to G values CF and RF 

"""
import pandas as pd
import networkx as nx

file_CF = 'C:/CAE/dummies/gnfe/physgnn/rescomp/dsallCF2_01.csv'
file_RF = 'C:/CAE/dummies/gnfe/physgnn/rescomp/dsallRF2_01.csv'
file_S = 'C:/CAE/dummies/gnfe/physgnn/rescomp/dsallS_01.csv'
G_adj =  'C:/CAE/dummies/gnfe/physgnn/preping/B2.adjlist'  # # grapooh defined by adjacency list 
G_ml =  'C:/CAE/dummies/gnfe/physgnn/preping/B2.graphml'  

x_CF = pd.read_csv(file_CF)
x_RF = pd.read_csv(file_RF)
y = pd.read_csv(file_S)
G = nx.read_adjlist(G_adj)
# %%
# https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.set_node_attributes.html
G = nx.read_graphml(G_ml)
