# -*- coding: utf-8 -*-
import torch as th
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append("..") # Adds higher directory to python modules path.
from load_dataset import load_dataset 
from modelling.modelsStore import SimpleNet, SAGE0
from modelling.modelsStore import ds_splitting
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import random 
import networkx as nx
import dgl
import matplotlib.pyplot as plt
import torch
import time
from dgl.nn import SAGEConv
from dgl.nn import GraphConv
import dgl.nn as dglnn
from dgl.dataloading import GraphDataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
gfds_name = 'Fibonacci'
pathRes  = 'fs0/'
methodID = 'LR' # FFNN,SAGE0,GCN0,
# json_file_name  = '../datasets/b3/'+gfds_name +'.json'
json_file_name  = '../datasets/fs/Fibonacci_spring.json'
# path_graph      = '../datasets/b2/'+gfds_name +'.adjlist'

D = load_dataset.dataset(json_file_name)
dkeys =D.getAvailableKeys()
X0 = D.selByKey('U.U1').T 
print(X0.max())
y = D.selByKey('S.Max. Prin').T 
scaler = MinMaxScaler()
df = pd.DataFrame(X0)
X0 = pd.DataFrame(scaler.fit_transform(df), columns=df.columns).values
print(X0.max())

# %%
experiments = [42,17,23,11,18,4,5,1,6,1212]
loss_fn = F.mse_loss
num_epochs = 1000
for experiment in experiments:
    # break
    experiment_name = f'{gfds_name}_{methodID}_{experiment}'
    torch.manual_seed(experiment)
    train_loader, test_loader = ds_splitting(X0,y)
   
   
    
    model = nn.Linear(X0.shape[1],y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    my_device = "cuda" if torch.cuda.is_available() else "cpu"    
    model = model.to(my_device)  
    
    losses = []
    losses_val = []
    time_elapsed = []
    epochs = []
    inputs = 'x'
    targets = 'y'
    t0 = time.time()    
    for epoch in range(num_epochs):        
        total_loss = 0.0
        batch_count = 0        
        for xb,yb in train_loader:            
            optimizer.zero_grad()
            # batch = batch.to(my_device)
            pred = model(xb.to(my_device))
            # pred = model(batch, batch.ndata[inputs].to(my_device))
            loss = loss_fn(pred, yb.to(my_device))
            # loss = loss_fn(pred, batch.ndata[targets].to(my_device))
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
        
        for xbv,ybv in test_loader:        
            # batch = batch.to(my_device)
            pred_val = model(xbv.to(my_device))
            # loss_val = loss_fn(pred_val, batch.ndata[targets].to(my_device))
            loss_val = loss_fn(pred_val, ybv.to(my_device))
            # total_val = ((pred -batch.ndata[targets])/(pred -batch.ndata[targets]).sum()).sum()
            total_val = ((pred_val -ybv.to(my_device))/(pred_val -ybv.to(my_device)).sum()).sum()
        if epoch % 5 == 1:
            print(f"validation accuracy = {loss_val} , {total_val}")
        losses_val.append(loss_val.detach())


    L =np.zeros([len(losses)])
    LV =np.zeros([len(losses)])
    for i in range(len(losses)):
        L[i] = losses[i].cpu().numpy()
        LV[i] = losses_val[i].cpu().numpy()
    np.save(pathRes + experiment_name + ".npy", 
        {"epochs": epochs, \
        "losses": L, \
        "losses_val": LV, \
        "time_elapsed": time_elapsed})  
    
    pathModel = pathRes + experiment_name + '.pt'
    torch.save(model.state_dict(),pathModel)

    plt.figure()
    plt.plot(L)
    plt.plot(LV) 
    plt.yscale("log")
    plt.savefig(pathRes + experiment_name + ".jpg")
    plt.close()