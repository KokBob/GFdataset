# -*- coding: utf-8 -*-
# https://github.com/KokBob/PhysGNN/blob/f527de0b2b9405de8bf9934e459bfa68ddb2ff38/models.py#L1
# https://www.youtube.com/watch?v=-UjytpbqX4A
# https://stackoverflow.com/questions/68202388/graph-neural-network-regression
# https://stackoverflow.com/questions/72849912/regression-case-for-graph-neural-network
# https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-1-d98dc93e7742
#%% LIBS
import dgl
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import networkx as nx
from sklearn.model_selection import train_test_split
import torch 
import torch as th
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
# %% PATHES
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
X = d.iloc[1:,[2,8,9]]
G = nx.read_graphml(G_ml)
seeding_magic_number = 42  
selectEach = 2
X= X.iloc[::selectEach, :]
y= y.iloc[::selectEach, :]
# x = th.tensor(X.values, dtype=torch.float)
# y = th.tensor(y.values, dtype=torch.float)
x = th.tensor(X.values[0:10], dtype=torch.float)
y = th.tensor(y.values[0:10], dtype=torch.float)
# %%
for nid, attr in G.nodes(data=True):
    print(attr)
    
# %%
import random 
ids = []
si = []
rf = []
for nid, attr in G.nodes(data=True):
    ids.append(nid)
    # si.append(attr['si'])
    rf.append(random.randint(30, 50))
# nodes = pd.DataFrame({'Id' : ids, 'si' : si, 'rf' : rf})
# %%

src = []
dst = []
weight = []
for u, v in G.edges():
    src.append(u)
    dst.append(v)
    weight.append(random.random())
edges = pd.DataFrame({'Src' : src, 'Dst' : dst, 'Weight' : weight})
# %%
g = dgl.from_networkx(G,)
# %%
u, v = G.edges()
# %%
eids = np.arange(G.number_of_edges())
eids = np.random.permutation(eids)
