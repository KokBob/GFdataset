# -*- coding: utf-8 -*-
"""
https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append("..") # Adds higher directory to python modules path.
from load_dataset import load_dataset 
from modelling.modelsStore import SimpleNet
import torch as th
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader # co je dataloade2?
import torch.nn.functional as F
import random 
import networkx as nx
import dgl
import matplotlib.pyplot as plt
import torch
import time

json_file_name  = '../datasets/fs/Fibonacci_spring.json'
path_graph      = '../datasets/fs/Fibonacci_spring.adjlist'

D = load_dataset.dataset(json_file_name)
dkeys =D.getAvailableKeys()
X0 = D.selByKey('U.U1').T # Plane
# X = D.selByKey('U.U3').T
y = D.selByKey('S.Max. Prin').T 
# y = D.selByKey('S.Max. Prin').T *10
# y = D.selByKey('RF.RF1').T
n=-1

XList_ = [1]
# yList = [0,1,2,3]
X = X0[:n,:]
X = X0[:n,XList_]
y = y[:n,:] # /
# y = y[:n,yList] 
# %%
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=.3)
# %%
Xtrain = th.tensor(Xtrain , dtype=th.float)
ytrain = th.tensor(ytrain, dtype=th.float)
train_ds = TensorDataset(Xtrain,ytrain) 
Xtest = th.tensor(Xtest , dtype=th.float)
ytest = th.tensor(ytest, dtype=th.float)
test_ds = TensorDataset(Xtest,ytest) 
# %%
train_dl = DataLoader(train_ds, batch_size = 10, shuffle=True)
test_dl  = DataLoader(test_ds, batch_size  = 10, shuffle= False)
# %%
def fit(num_epochs, model, loss_fn, opt):
    running_loss = 0.
    last_loss = 0.
    vloss_ = []
    for epoch in range(num_epochs):
        running_vloss = 0.0
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
        print('Training loss: ', loss_fn(model(xb), yb))
        for i, vdata in enumerate(test_dl):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
            vl = float(running_vloss.detach())
            vloss_.append(vl)
        # Gather data and report
        running_loss += loss.item()
        # if i % 1000 == 999:
            # last_loss = running_loss / 1000 # loss per batch
            # print('  batch {} loss: {}'.format(i + 1, last_loss))
            # tb_x = epoch_index * len(training_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            # running_loss = 0.
    # return running_vloss
    return vloss_
# %%
modelF = SimpleNet(X.shape[1],y.shape[1])
# modelF = nn.Linear(X.shape[1],y.shape[1])
optF = th.optim.SGD(modelF.parameters(), 1e-3) # 1e-3 Smax 
# optF = th.optim.SGD(modelF.parameters(), 1e-7) # plane 1e-7 
loss_fnF = F.mse_loss
running_vloss = fit(1000, modelF, loss_fnF, optF)
plt.plot(running_vloss)
# %%
# option 1: DGL will than create only one directional mesh based graph
G = nx.read_adjlist(path_graph).to_directed() 
g = dgl.from_networkx(G)

d_ = D.Dataset
# %%
import torch as th
i = 0
graphs = []
for _i in D.Dataset:
    d_i = D.Dataset[_i]
    frame = _i.split('\\')[1]
    # g_ = g
    g_ = dgl.from_networkx(G)
    g_.name = frame
    
    g_.ndata['x'] 		= th.tensor(X0[i,:],dtype = th.float)
    g_.ndata['y'] 		= th.tensor(y[i,:],dtype = th.float)
    # break
    graphs.append(g_)
    i +=1
    # break
    # print(_i.split('\\')[1])
    print(frame)
# %%
random.seed(42)
# graphs_shuffled = random.shuffle(graphs)
random.shuffle(graphs)
for go in graphs: print(go.name)
# %% poznamky pro list
from dgl.nn import SAGEConv
from dgl.nn import GraphConv
import dgl.nn as dglnn
from dgl.dataloading import GraphDataLoader
import torch.nn as nn
import torch.nn.functional as F
# Define a graph convolutional network
class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hid_feats, norm='both', weight=True, bias=True)
        self.conv2 = dglnn.SAGEConv(hid_feats, out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
    
class SAGE(nn.Module):
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
def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
train_loader = GraphDataLoader(graphs[:70],shuffle=True, )
test_loader = GraphDataLoader(graphs[70:],shuffle=False, ) 
# %%
# train_loader = GraphDataLoader(graphs_shuffled[:50],shuffle=True, batch_size=8)
# test_loader = GraphDataLoader(graphs_shuffled[50:],shuffle=False, batch_size=1) 
# %%
# train_loader = GraphDataLoader(train_dataset, batch_size=8)
# test_loader = GraphDataLoader(test_dataset, batch_size=1)    
for batch in train_loader: break    
in_channels = batch.ndata['y'].shape[0]
# in_channels = batch.ndata['S.mises'].shape[0]
# out_channels = batch.ndata['S.mises'].shape[1]

# %%

# net = GraphSAGE(5, 16, 2)
model = SAGE(in_channels, in_channels, in_channels)
# model = GCN(in_channels, in_channels, in_channels)

# loss_fn = torch.nn.BCELoss()
loss_fn = F.mse_loss
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # obscurly running back 
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) # converging quite monotonically but too slowly 
# %
num_epochs =1000
my_device = "cuda" if torch.cuda.is_available() else "cpu"    
model = model.to(my_device)  
losses = []
losses_val = []
time_elapsed = []
epochs = []
inputs = 'x'
# inputs = 'U.Magnitude'
targets = 'y'
t0 = time.time()    
for epoch in range(num_epochs):        
    total_loss = 0.0
    batch_count = 0        
    for batch in train_loader:            
            optimizer.zero_grad()
            batch = batch.to(my_device)
            pred = model(batch, batch.ndata[inputs].to(my_device))
            loss = loss_fn(pred, batch.ndata[targets].to(my_device))
            loss.backward()
            optimizer.step()            
            total_loss += loss.detach()
            batch_count += 1        
            mean_loss = total_loss / batch_count
    losses.append(mean_loss)
    epochs.append(epoch)
    time_elapsed.append(time.time() - t0)        
    # if epoch % 100 == 0:
    if epoch % 5 == 1:
        print(f"loss at epoch {epoch} = {mean_loss}")    # get test accuracy score
    num_correct = 0.
    num_total = 0.
    model.eval()    
    
    for batch in test_loader:        
        batch = batch.to(my_device)
        pred_val = model(batch, batch.ndata[inputs])
        loss_val = loss_fn(pred_val, batch.ndata[targets].to(my_device))
        total_val = ((pred -batch.ndata[targets])/(pred -batch.ndata[targets]).sum()).sum()
    if epoch % 5 == 1:
        print(f"validation accuracy = {loss_val} , {total_val}")
    losses_val.append(loss_val.detach())
        # num_correct += (pred.round() ==  batch.ndata[targets].to(my_device)).sum()
        # num_total += pred.shape[0] * pred.shape[1]        
        # np.save("dgl.npy", 
            # {"epochs": epochs, \
            # "losses": losses, \
            # "time_elapsed": time_elapsed})    
        # %%
print(f"time_elapsed = {time_elapsed[-1]}")     
# %
import matplotlib.pyplot as plt
plt.figure()
plt.plot(losses)
plt.plot(losses_val)
plt.yscale("log")


# %%
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
# %
pathModel = 'fs_sage_01.pt'
torch.save(model.state_dict(),pathModel)


# %%
# modelT = GCN(in_channels, in_channels, in_channels)
modelT = SAGE(in_channels, in_channels, in_channels)
modelT.load_state_dict(torch.load(pathModel))
modelT.eval()
# %%
for batch in test_loader:        
    batch = batch.to(my_device)
    pred_val = modelT(batch, batch.ndata[inputs])
    loss_val = loss_fn(pred_val, batch.ndata[targets].to(my_device))
    total_val = ((pred -batch.ndata[targets])/(pred -batch.ndata[targets]).sum()).sum()
# if epoch % 5 == 1:
    print(f"validation accuracy = {loss_val} , {total_val}")
losses_val.append(loss_val.detach())
    # num_correct += (pred.round() ==  batch.ndata[targets].to(my_device)).sum()
    # num_total += pred.shape[0] * pred.shape[1]        
    # np.save("dgl.npy", 
        # {"epochs": epochs, \
        # "losses": losses, \
        # "time_elapsed": time_elapsed})    