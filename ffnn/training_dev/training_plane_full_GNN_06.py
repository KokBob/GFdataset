# -*- coding: utf-8 -*-
training_execute = False
# training_execute = True
import sys
import pandas as pd
sys.path.append("..") 
from load_dataset import load_dataset 
from modelling.modelsStore import GCN0
from modelling.modelsStore import SAGE0
from modelling.preprocessing import *
import modelling.experimentationing as exps
import torch
from sklearn.preprocessing import MinMaxScaler
GFDS = load_dataset.Plane()
gfds_name   = GFDS.gfds_name
pathRes     = GFDS.pathRes
D           = GFDS.D

X0 = GFDS.X0
y = GFDS.y
G = GFDS.G
df = pd.DataFrame(X0)
dfy = pd.DataFrame(y)

scaler = MinMaxScaler()
df = pd.DataFrame(X0)
X0 = pd.DataFrame(scaler.fit_transform(df), columns=df.columns).values

graphs = graphs_preparation(D, G, X0, y)

# experiments_IDs = [42,17] # unscaled
experiments_IDs = [28,69] # scaledX
num_epochs      = 500
# %% training loop pro gcn
if training_execute: 
    methodID    = 'GCN_RF2' 
    MODEL       = GCN0
    for experiment_number in experiments_IDs:
        
        exp_ = exps.Experiment_Graph(gfds_name, methodID, experiment_number, graphs)
        pathModel = pathRes + exp_.experiment_name + '.pt'
        exp_.training_preparation(MODEL)
        # training 
        exp_.training_run(num_epochs)
        exp_.validate()
        torch.save(exp_.beast,pathModel)
    # %% training loop pro sage
    methodID    = 'SAGE_RF2' 
    MODEL       = SAGE0
    for experiment_number in experiments_IDs:
        
        exp_ = exps.Experiment_Graph(gfds_name, methodID, experiment_number, graphs)
        pathModel = pathRes + exp_.experiment_name + '.pt'
        exp_.training_preparation(MODEL)
        # training 
        exp_.training_run(num_epochs)
        exp_.validate()
        # torch.save(exp_.model.state_dict(),pathModel)
        torch.save(exp_.beast,pathModel)
else: pass
# %%
    # loading from trained 
# experiment_number = 28
experiment_number = 23
methodID    = 'GCN_RF2' 
MODEL       = GCN0
exp_ = exps.Experiment_Graph(gfds_name, methodID, experiment_number, graphs)
pathModel = pathRes + exp_.experiment_name + '.pt'
exp_.training_preparation(MODEL)
exp_.model.load_state_dict(torch.load(pathModel))
exp_.model.eval()
exp_.validate()
exp_.validate_plot()
    
