# -*- coding: utf-8 -*-
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append("..") 
from load_dataset import load_dataset 
from modelling.modelsStore import GCN0
from modelling.preprocessing import *
from modelling.experimentationing import *
import torch.nn.functional as F
import random 
from sklearn.preprocessing import MinMaxScaler
import torch
import time
from dgl.dataloading import GraphDataLoader

GFDS = load_dataset.Fibonacci()
# GFDS = load_dataset.Beam2D('RF.RF2')
gfds_name = GFDS.gfds_name
pathRes  = GFDS.pathRes
methodID = 'GCN_RF3' # FFNN,SAGE0,GCN0,
MODEL = GCN0
D = GFDS.D
X0 = GFDS.X0
y = GFDS.y
G = GFDS.G
print(X0.max())
scaler = MinMaxScaler()
df = pd.DataFrame(X0)
X0 = pd.DataFrame(scaler.fit_transform(df), columns=df.columns).values
print(X0.max())
# %%
graphs = graphs_preparation(D, G, X0, y)
# %%
# experiments_IDs = [42,17,23,11,18,4,5,1,6,1212]
experiments_IDs = [18,4,5,1,6,1212]
loss_fn = F.mse_loss
num_epochs = 500
for experiment_number in experiments_IDs:
    
    experiment_name = f'{gfds_name}_{methodID}_{experiment_number}'
    random.seed(experiment_number)
    random.shuffle(graphs)

    split_number = int(len(graphs)*.7)
    train_loader = GraphDataLoader(graphs[:split_number],shuffle=True, )
    test_loader = GraphDataLoader(graphs[split_number:],shuffle=False, ) 
    
    for batch in train_loader: break    
    in_channels = batch.ndata['y'].shape[0]
    
    
    
    model = MODEL(in_channels, in_channels, in_channels)
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
        total_loss_val = 0.0
        batch_count_val = 0 
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
        if epoch % 5 == 1:
            print(f"loss at epoch {epoch} = {mean_loss}")    # get test accuracy score
        num_correct = 0.
        num_total = 0.
        model.eval()    
        
        for batch in test_loader:        
            batch = batch.to(my_device)
            pred_val = model(batch, batch.ndata[inputs])
            loss_val = loss_fn(pred_val, batch.ndata[targets].to(my_device))
            total_loss_val += loss_val.detach()
            batch_count_val += 1        
            mean_loss_val = total_loss_val / batch_count_val
            total_val = ((pred -batch.ndata[targets])/(pred -batch.ndata[targets]).sum()).sum()
        if epoch % 5 == 1:
            print(f"validation accuracy = {loss_val} , {total_val}")
            # print(f"validation accuracy = {mean_loss_val} , {total_val}") # zatim neaplikovat
        # losses_val.append(loss_val.detach())
        losses_val.append(mean_loss_val)
    pathModel = pathRes + experiment_name + '.pt'
    torch.save(model.state_dict(),pathModel)
    experiment_evaluation(experiment_name,
                     pathRes,
                     model,
                     epochs,
                     time_elapsed,
                     losses, 
                     losses_val)
from modelling.modelsStore import SAGE0
methodID = 'SAGE_RF2' # FFNN,SAGE0,GCN0,
MODEL = SAGE0
for experiment_number in experiments_IDs:
    
    experiment_name = f'{gfds_name}_{methodID}_{experiment_number}'
    random.seed(experiment_number)
    random.shuffle(graphs)

    split_number = int(len(graphs)*.7)
    train_loader = GraphDataLoader(graphs[:split_number],shuffle=True, )
    test_loader = GraphDataLoader(graphs[split_number:],shuffle=False, ) 
    
    for batch in train_loader: break    
    in_channels = batch.ndata['y'].shape[0]
    
    
    
    model = MODEL(in_channels, in_channels, in_channels)
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
        total_loss_val = 0.0
        batch_count_val = 0 
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
        if epoch % 5 == 1:
            print(f"loss at epoch {epoch} = {mean_loss}")    # get test accuracy score
        num_correct = 0.
        num_total = 0.
        model.eval()    
        
        for batch in test_loader:        
            batch = batch.to(my_device)
            pred_val = model(batch, batch.ndata[inputs])
            loss_val = loss_fn(pred_val, batch.ndata[targets].to(my_device))
            total_loss_val += loss_val.detach()
            batch_count_val += 1        
            mean_loss_val = total_loss_val / batch_count_val
            total_val = ((pred -batch.ndata[targets])/(pred -batch.ndata[targets]).sum()).sum()
        if epoch % 5 == 1:
            print(f"validation accuracy = {loss_val} , {total_val}")
            # print(f"validation accuracy = {mean_loss_val} , {total_val}") # zatim neaplikovat
        # losses_val.append(loss_val.detach())
        losses_val.append(mean_loss_val)
    pathModel = pathRes + experiment_name + '.pt'
    torch.save(model.state_dict(),pathModel)
    experiment_evaluation(experiment_name,
                     pathRes,
                     model,
                     epochs,
                     time_elapsed,
                     losses, 
                     losses_val)
