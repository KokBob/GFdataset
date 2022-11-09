# -*- coding: utf-8 -*-
"""
https://pytorch.org/docs/stable/nn.functional.html
https://stackoverflow.com/questions/25055712/pandas-every-nth-row
"""
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
file_CF  = '../rescomp/dsallCF2_01.csv'
file_RF  = '../rescomp/dsallRF2_01.csv'
file_S   = '../rescomp/dsallS_01.csv'
G_adj    = '../preping/B2.adjlist'  # # grapooh defined by adjacency list 
G_ml     = '../preping/B2.graphml'  
# %% reading data
y = pd.read_csv(file_S).T
y = y.drop(index='Unnamed: 0')
x_CF = pd.read_csv(file_CF)
x_CF = x_CF.T 
x_RF = pd.read_csv(file_RF)
x_RF = x_RF.T
d= pd.concat([x_CF,x_RF], axis = 1)
X = d.iloc[1:,[2,8,9]] #
# X = d.iat[1,[2,8,9]]
G = nx.read_graphml(G_ml)
seeding_magic_number = 42  
# %%
# Xred= X[X.index % 3 == 0]  # Selects every 3rd row starting from 0
Xred= X.iloc[::50, :]
yred=y.iloc[::50, :]
# %% Splitting and shuffling
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seeding_magic_number)
# %% set attribute to G
import networkx as nx
X2G = X.iloc[0].values
# nx.set_node_attributes(G, X['Label'], "Node Labels FEM")
Gd = dgl.from_networkx(G) # from networkx
club = th.tensor(X2G, dtype=th.int64)
# %%


