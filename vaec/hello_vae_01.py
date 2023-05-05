# -*- coding: utf-8 -*-
# GFDS = load_dataset.Beam3D()
# gfds_name   = GFDS.gfds_name
# pathRes     = GFDS.pathRes
# D           = GFDS.D

# X0 = GFDS.X0
# y = GFDS.y
# G = GFDS.G
# df = pd.DataFrame(X0)
# dfy = pd.DataFrame(y)

import torch
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from modelling.modelsStore import SimpleNet, SAGE0
from modelling.modelsStore import ds_splitting
from load_dataset import load_dataset 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from evaluation import evaluate_model
gfds_name = 'Beam2D'
pathRes  = './b2/'
methodID = 'FN_RF2' # FFNN,SAGE0,GCN0,
json_file_name  = '../datasets/b2/'+gfds_name +'.json'
path_graph      = '../datasets/b2/'+gfds_name +'.adjlist'

D = load_dataset.dataset(json_file_name)
dkeys =D.getAvailableKeys()
X0 = D.selByKey('RF.RF2').T 
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
DR = {}
# fig, axs = plt.subplots(2, 2)
for experiment in experiments:
    # break
    DR[experiment] = {}
    dr_ = DR[experiment] 
    experiment_name = f'{gfds_name}_{methodID}_{experiment}'
    torch.manual_seed(experiment)
    train_loader, test_loader = ds_splitting(X0,y)    
    break
