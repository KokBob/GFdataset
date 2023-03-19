# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append("..") # Adds higher directory to python modules path.
from load_dataset import load_dataset 
from modelling.modelsStore import GCN0
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
from dgl.nn import SAGEConv
from dgl.nn import GraphConv
import dgl.nn as dglnn
from dgl.dataloading import GraphDataLoader
import torch.nn as nn
import torch.nn.functional as F
gfds_name = 'Beam3D'
pathRes  = 'b3/'
methodID = 'GC0' # FFNN,SAGE0,GCN0, 
json_file_name  = '../datasets/b3/Beam3D.json'
path_graph      = '../datasets/b3/Beam3D.adjlist'

D = load_dataset.dataset(json_file_name)
dkeys =D.getAvailableKeys()
X0 = D.selByKey('U.U1').T 
print(X0.max())
y = D.selByKey('S.Max. Prin').T 
G = nx.read_adjlist(path_graph).to_directed() 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df = pd.DataFrame(X0)
X0 = pd.DataFrame(scaler.fit_transform(df), columns=df.columns).values
print(X0.max())

# %%
import torch as th
i = 0
graphs = []
for _i in D.Dataset:
    try:
        d_i = D.Dataset[_i]
        frame = _i.split('\\')[1]
        g_ = dgl.from_networkx(G)
        g_.name = frame
        g_.ndata['x'] 		= th.tensor(X0[i,:],dtype = th.float)
        g_.ndata['y'] 		= th.tensor(y[i,:],dtype = th.float)
        graphs.append(g_)
        i +=1
    except: pass
# %%
experiments = [42,17,23,11,18,4,5,1,6,1212]
loss_fn = F.mse_loss

for experiment in experiments:
    # break
    experiment_name = f'{gfds_name}_{methodID}_{experiment}'
    random.seed(experiment)
    random.shuffle(graphs)

   
    train_loader = GraphDataLoader(graphs[:70],shuffle=True, )
    test_loader = GraphDataLoader(graphs[70:],shuffle=False, ) 
    for batch in train_loader: break    
    in_channels = batch.ndata['y'].shape[0]
    
    
    model = GCN0(in_channels, in_channels, in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    my_device = "cuda" if torch.cuda.is_available() else "cpu"    
    num_epochs =500
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