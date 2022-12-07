# -*- coding: utf-8 -*-
# https://github.com/KokBob/PhysGNN/blob/f527de0b2b9405de8bf9934e459bfa68ddb2ff38/models.py#L1
# https://www.youtube.com/watch?v=-UjytpbqX4A
# https://stackoverflow.com/questions/68202388/graph-neural-network-regression
# https://stackoverflow.com/questions/72849912/regression-case-for-graph-neural-network
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html
# https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-1-d98dc93e7742
# https://pytorch.org/docs/stable/tensorboard.html
# https://colab.research.google.com/github/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial7/GNN_overview.ipynb
# https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/directional_GSN/main.py
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
node_feats = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
adj_matrix = torch.Tensor([[[1, 1, 0, 0],
                            [1, 1, 1, 1],
                            [0, 1, 1, 1],
                            [0, 1, 1, 1]]])
print("Node features:\n", node_feats)
print("\nAdjacency matrix:\n", adj_matrix)
# %%
class GCNLayer(nn.Module):
    
    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections. 
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats
# %%
layer = GCNLayer(c_in=2, c_out=2)
layer.projection.weight.data = torch.Tensor([[1., 0.], [0., 1.]])
layer.projection.bias.data = torch.Tensor([0., 0.])
# %
with torch.no_grad():
    out_feats = layer(node_feats, adj_matrix)

print("Adjacency matrix", adj_matrix)
print("Input features", node_feats)
print("Output features", out_feats)
# for nid, attr in G.nodes(data=True):
#     print(attr)
    
# # %%
# import random 
# ids = []
# si = []
# rf = []
# for nid, attr in G.nodes(data=True):
#     ids.append(nid)
#     # si.append(attr['si'])
#     rf.append(random.randint(30, 50))
# # nodes = pd.DataFrame({'Id' : ids, 'si' : si, 'rf' : rf})
# # %%
# src = []
# dst = []
# weight = []
# for u, v in G.edges():
#     src.append(u)
#     dst.append(v)
#     weight.append(random.random())
# edges = pd.DataFrame({'Src' : src, 'Dst' : dst, 'Weight' : weight})
# # %%
# g = dgl.from_networkx(G,)
# # %%
# # u, v = G.edges()
# # %%
# eids = np.arange(G.number_of_edges())
# eids = np.random.permutation(eids)
# from torch_geometric.datasets import Planetoid
# from torch_geometric.transforms import NormalizeFeatures

# ind.cora.x
# Using existing file ind.cora.x
# Using existing file ind.cora.tx
# https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.allx
# Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.y
# Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.ty
# Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.ally
# Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.graph
# Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.test.index

