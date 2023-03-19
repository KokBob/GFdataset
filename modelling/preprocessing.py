# -*- coding: utf-8 -*-
import dgl
import torch as th
from sklearn.preprocessing import MinMaxScaler
def dataset_select(dataset_identifier):
    # dataset_identifier
    # "keys: B2, B3, FS, PL,
    pass
def graphs_preparation(dataset_class,initial_graph, input_features, output_features):
    D = dataset_class
    G = initial_graph
    X0 = input_features
    y = output_features
    i = 0
    graphs = []
    for _i in D.Dataset:
        try:
            # d_i = D.Dataset[_i]
            frame = _i.split('\\')[1]
            g_ = dgl.from_networkx(G)
            g_.name = frame
            g_.ndata['x'] 		= th.tensor(X0[i,:],dtype = th.float)
            g_.ndata['y'] 		= th.tensor(y[i,:],dtype = th.float)
            graphs.append(g_)
            i +=1
        except: pass
    return graphs