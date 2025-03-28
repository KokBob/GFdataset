# -*- coding: utf-8 -*-

import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
# https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
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
G = nx.read_graphml(G_ml)
seeding_magic_number = 42  
X= X.iloc[::50, :]
y=y.iloc[::50, :]
# %% Splitting and shuffling
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seeding_magic_number)
# %%
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
 
net = nn.Sequential(
      nn.Linear(3, 4),
      nn.Sigmoid(),
      nn.Linear(4, 1),
      nn.Sigmoid()
      ).to(device)

print(net)
