# -*- coding: utf-8 -*-
import torch as th
import torch.nn as nn
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import split_dataset
import torch.nn.functional as F
import dgl.nn as dglnn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader # co je dataloade2?


def ds_splitting(X,y):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=.3)
    
    Xtrain = th.tensor(Xtrain , dtype=th.float)
    ytrain = th.tensor(ytrain, dtype=th.float)
    
    Xtest = th.tensor(Xtest , dtype=th.float)
    ytest = th.tensor(ytest, dtype=th.float)
    
    train_ds = TensorDataset(Xtrain,ytrain) 
    test_ds = TensorDataset(Xtest,ytest) 
    
    train_dl = DataLoader(train_ds, batch_size = 10, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size = 10, shuffle= True)
    return train_dl, test_dl
class SAGE0(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
class GCN0(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hid_feats, norm='both', weight=True, bias=True)
        # self.conv12 = dglnn.GraphConv(in_feats, hid_feats, norm='both', weight=True, bias=True)
        self.conv2 = dglnn.SAGEConv(hid_feats, out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        # h = self.conv12(graph, inputs)
        # h = F.relu(h)
        h = self.conv2(graph, h)
        return h
    
class LR0(nn.Module):
    # input_size = 
    # output_size = 
    # Initialize the layers
    def __init__(self, input_size, output_size ):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear(x)
        return x    
class SimpleNet(nn.Module):
    # input_size = 
    # output_size = 
    # Initialize the layers
    def __init__(self, input_size, output_size ):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 150)
        # self.linear1 = nn.Linear(input_size, input_size * 2)
        self.act1 = nn.ReLU() # Activation function
        # self.act2 = nn.Sigmoid() # Activation function
        # self.linear2 = nn.Linear(150, 2)
        self.linear2 = nn.Linear(150, output_size)
        # self.linear2 = nn.Linear(input_size*2, output_size)
        
        
        
    
    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        # x = self.act2(x)
        x = self.linear2(x)
        return x
    
# Define a graph convolutional network
class GraphNetwork(th.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.gc1 = GraphConv(in_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = torch.relu(x)
        x = self.gc2(x, edge_index)
        return x
    
class GraphConvNet(nn.Module):    
    def __init__(self, in_channels=3, out_channels=6):
        super(GraphConvNet, self).__init__()        
        self.gcn_0 = GraphConv(in_channels, 64,
                               allow_zero_in_degree=True)        
        self.gcn_h1 = GraphConv(64, 64, allow_zero_in_degree=True)
        self.gcn_h2 = GraphConv(64, 64, allow_zero_in_degree=True)
        self.gcn_h3 = GraphConv(64, 64, allow_zero_in_degree=True)
        self.gcn_h4 = GraphConv(64, 64, allow_zero_in_degree=True)
        self.gcn_h5 = GraphConv(64, 64, allow_zero_in_degree=True)
        self.gcn_h6 = GraphConv(64, 64, allow_zero_in_degree=True)        
        self.gcn_out = GraphConv(64, out_channels,  
                                 allow_zero_in_degree=True)    
    def forward(self, g, features):        
        x = torch.relu(self.gcn_0(g, features))        
        x = torch.relu(self.gcn_h1(g, x))
        x = torch.relu(self.gcn_h2(g, x))
        x = torch.relu(self.gcn_h3(g, x))
        x = torch.relu(self.gcn_h4(g, x))
        x = torch.relu(self.gcn_h5(g, x))
        x = torch.relu(self.gcn_h6(g, x))        
        x = torch.dropout(x, p=0.25, train=self.training)
        x = self.gcn_out(g, x)        
        x = torch.sigmoid(x)        
        return x
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_outputs):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(3, 150, 'mean')
        self.conv2 = SAGEConv(h_feats, num_classes, 'mean')
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
