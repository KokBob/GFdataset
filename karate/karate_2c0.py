#!usr/bin/env python
#encoding:utf-8
from __future__ import division
 


"""
https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=auto&commit=8fd0bbf9aca2bfcb95074e4124e01a2b074be300&device=unknown&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f64676c61692f57575732302d48616e64732d6f6e2d5475746f7269616c2f386664306262663961636132626663623935303734653431323465303161326230373462653330302f62617369635f7461736b732f335f6c696e6b5f707265646963742e6970796e62&logged_in=false&nwo=dglai%2FWWW20-Hands-on-Tutorial&path=basic_tasks%2F3_link_predict.ipynb&platform=android&repository_id=216741794&repository_type=Repository&version=99
https://notebooks.githubusercontent.com/view/ipynb?azure_maps_enabled=false&browser=chrome&color_mode=auto&commit=8fd0bbf9aca2bfcb95074e4124e01a2b074be300&device=unknown&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f64676c61692f57575732302d48616e64732d6f6e2d5475746f7269616c2f386664306262663961636132626663623935303734653431323465303161326230373462653330302f62617369635f7461736b732f325f676e6e2e6970796e62&enterprise_enabled=false&logged_in=false&nwo=dglai%2FWWW20-Hands-on-Tutorial&path=basic_tasks%2F2_gnn.ipynb&platform=android&repository_id=216741794&repository_type=Repository&version=102
"""


import dgl
import numpy as np
import networkx as nx
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import itertools
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import random
import pandas as pd


# %%
g = nx.karate_club_graph().to_undirected().to_directed()
ids = []
clubs = []
ages = []
for nid, attr in g.nodes(data=True):
    ids.append(nid)
    clubs.append(attr['club'])
    ages.append(random.randint(30, 50))
nodes = pd.DataFrame({'Id' : ids, 'Club' : clubs, 'Age' : ages})
print(nodes)
src = []
dst = []
weight = []
for u, v in g.edges():
    src.append(u)
    dst.append(v)
    weight.append(random.random())
edges = pd.DataFrame({'Src' : src, 'Dst' : dst, 'Weight' : weight})
print(edges)
g = dgl.graph((src, dst))

# %%
nodes_data = nodes
# nodes_data = pd.read_csv('data/nodes.csv')
club = nodes_data['Club'].to_list()
# Convert to categorical integer values with 0 for 'Mr. Hi', 1 for 'Officer'.
club = th.tensor([c == 'Officer' for c in club], dtype=th.int64)
club_onehot = th.nn.functional.one_hot(club, club.max()+1)
print(club_onehot)
# %%
# Use `g.ndata` like a normal dictionary
g.ndata.update({'club' : club, 'club_onehot' : club_onehot})
labels = g.ndata['club']
# labeled_nodes = [0, 33]
labeled_nodes = [0, 33]
print('Labels', labels[labeled_nodes])
# %%

# ----------- 1. node features -------------- #
node_embed = nn.Embedding(g.number_of_nodes(), 5)  # Every node has an embedding of size 5.
inputs = node_embed.weight                         # Use the embedding weight as the node features.
nn.init.xavier_uniform_(inputs)
print(inputs)
# %%
# Split edge set for training and testing
u, v = g.edges()
eids = np.arange(g.number_of_edges())
eids = np.random.permutation(eids)
test_pos_u, test_pos_v = u[eids[:50]], v[eids[:50]]
train_pos_u, train_pos_v = u[eids[50:]], v[eids[50:]]
# %%
import scipy.sparse as sp
# Find all negative edges and split them for training and testing
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(34)
neg_u, neg_v = np.where(adj_neg != 0)
neg_eids = np.random.choice(len(neg_u), 200)
test_neg_u, test_neg_v = neg_u[neg_eids[:50]], neg_v[neg_eids[:50]]
train_neg_u, train_neg_v = neg_u[neg_eids[50:]], neg_v[neg_eids[50:]]
# %%
# Create training set.
train_u = torch.cat([torch.as_tensor(train_pos_u), torch.as_tensor(train_neg_u)])
train_v = torch.cat([torch.as_tensor(train_pos_v), torch.as_tensor(train_neg_v)])
train_label = torch.cat([torch.zeros(len(train_pos_u)), torch.ones(len(train_neg_u))])

# Create testing set.
test_u = torch.cat([torch.as_tensor(test_pos_u), torch.as_tensor(test_neg_u)])
test_v = torch.cat([torch.as_tensor(test_pos_v), torch.as_tensor(test_neg_v)])
test_label = torch.cat([torch.zeros(len(test_pos_u)), torch.ones(len(test_neg_u))])
# %%
from dgl.nn import SAGEConv
from dgl.nn import ChebConv
from dgl.nn import SGConv
# Parameters https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.conv.ChebConv.html
# in_feats (int) – Dimension of input features; i.e, the number of dimensions of h(l)i.

# out_feats (int) – Dimension of output features h(l+1)i.

# k (int) – Chebyshev filter size K.

# activation (function, optional) – Activation function. Default ReLu.

# bias (bool, optional) – If True, adds a learnable bias to the output. Default: True.


# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        
        self.conv1 = GraphConv(in_feats, h_feats, norm='both', weight=True, bias=True)
        # self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        # self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        
        
        self.conv2 = GraphConv(h_feats, h_feats, 'both')
        # self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        # self.conv2 = ChebConv(h_feats, h_feats, 2)
        
        # self.conv3 = SAGEConv(h_feats, h_feats, 'mean')
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        # h = self.conv3(g, h)
        return h
    
# Create the model with given dimensions 
# input layer dimension: 5, node embeddings
# hidden layer dimension: 16
net = GraphSAGE(5, 16)
# %%
# ----------- 1. node features -------------- #
node_embed = nn.Embedding(g.number_of_nodes(), 5)  # Every node has an embedding of size 5.
inputs = node_embed.weight                         # Use the embedding weight as the node features.
nn.init.xavier_uniform_(inputs)
# ----------- 3. set up loss and optimizer -------------- #
# in this case, loss will in training loop
optimizer = torch.optim.Adam(itertools.chain(net.parameters(), node_embed.parameters()), lr=0.01)

# ----------- 4. training -------------------------------- #
all_logits = []
for e in range(1000):
    # forward
    logits = net(g, inputs)
    pred = torch.sigmoid((logits[train_u] * logits[train_v]).sum(dim=1))
    
    # compute loss
    loss = F.binary_cross_entropy(pred, train_label)
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    all_logits.append(logits.detach())
    
    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))
# %%
# ----------- 5. check results ------------------------ #
pred = torch.sigmoid((logits[test_u] * logits[test_v]).sum(dim=1))
print('Accuracy', ((pred >= 0.5) == test_label).sum().item() / len(pred))
