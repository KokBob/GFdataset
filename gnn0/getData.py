# -*- coding: utf-8 -*-
import pandas as pd
import torch
import networkx as nx
def getCRdata():
    # %% PATHES
    file_CF  = '../rescomp/dsallCF2_01.csv'
    file_RF  = '../rescomp/dsallRF2_01.csv'
    file_S   = '../rescomp/dsallS_01.csv'
    G_adj    = '../preping/B2.adjlist'  # # grapooh defined by adjacency list 
    G_ml     = '../preping/B2.graphml'  
    #reading data
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
    # selectEach = 2
    # X= X.iloc[::selectEach, :]
    # y= y.iloc[::selectEach, :]
    x = torch.tensor(X.values, dtype=torch.float)
    y = torch.tensor(y.values, dtype=torch.float)
    # x = th.tensor(X.values[0:10], dtype=torch.float)
    # y = th.tensor(y.values[0:10], dtype=torch.float)
    # return {'G': G, 'x': x, 'y': y}
    return G, x, y
