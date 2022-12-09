# -*- coding: utf-8 -*-
#https://medium.com/@khang.pham.exxact/pytorch-geometric-vs-deep-graph-library-626ff1e802
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv
from dgl.data import PPIDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import split_dataset
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
# %%
num_epochs = 10000
lr = 1e-3
my_seed = 42    
# dataset = PPIDataset()
# %%
# data.dgl.ai/dataset/ppi.zip
feats_test = np.load('./ppi/test_feats.npy')
labels_test = np.load('./ppi/test_labels.npy')
