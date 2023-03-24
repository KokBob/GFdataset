# -*- coding: utf-8 -*-
"""

"""
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
gfds_name = 'Beam2D'
pathRes  = '../ffnet/b2/'
# methodID = 'LR' # FFNN,SAGE0,GCN0,
methodID = 'FN' # FFNN,SAGE0,GCN0,
json_file_name  = '../datasets/b2/'+gfds_name +'.json'
path_graph      = '../datasets/b2/'+gfds_name +'.adjlist'

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
# experiments = [42,17,23,11,18,4,5,1,6,1212]
experiments = [17,23,11,18,4,5,1,6,1212]
loss_fn = F.mse_loss
num_epochs = 1000
for experiment in experiments:
    # break
    experiment_name = f'{gfds_name}_{methodID}_{experiment}'
    torch.manual_seed(experiment)
    train_loader, test_loader = ds_splitting(X0,y)    
    model = SimpleNet(X0.shape[1],y.shape[1])
    # model = nn.Linear(X0.shape[1],y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    my_device = "cuda" if torch.cuda.is_available() else "cpu"    
    model = model.to(my_device)  
    pathModel = pathRes + experiment_name + '.pt'
    break

# pathModel = pathRes + experiment_name + '.pt'

# torch.load(model.state_dict(),pathModel)
# Model class must be defined somewhere
model.load_state_dict(torch.load(pathModel))
# model = torch.load(pathModel)
model.eval()
# %%
for xb,yb in train_loader:            
    # optimizer.zero_grad()
    # batch = batch.to(my_device)
    pred = model(xb.to(my_device))
# %%
for xbv,ybv in test_loader:        
    # batch = batch.to(my_device)
    pred_val = model(xbv.to(my_device))
    # loss_val = loss_fn(pred_val, batch.ndata[targets].to(my_device))
    loss_val = loss_fn(pred_val, ybv.to(my_device))
    # total_val = ((pred -batch.ndata[targets])/(pred -batch.ndata[targets]).sum()).sum()
    total_val = ((pred_val -ybv.to(my_device))/(pred_val -ybv.to(my_device)).sum()).sum()
    # %
    for i in range(ybv.shape[0]):   
        y       =   ybv.to(my_device)[i].cpu().numpy()
        y_hat   =   pred_val[i].cpu().detach().numpy()
        err = y - y_hat
        # %
        # colors = np.random.rand(len(err))
        # plt.scatter(y, y_hat,  c=err, alpha=0.5)
        plt.scatter(y_hat, err,  c=err, alpha=0.5)
        # plt.hist(err, bins = 20) 
        # plt.scatter(y, y_hat, s=err*0.001, c=err, alpha=0.5)
        # plt.scatter(y, y_hat)
        # sm.qqplot(err/, line ='45')
        # sm.qqplot(err/err.max(), line ='45')
        plt.show()
    # break
# %% 

# y       =   ybv.to(my_device)[0].cpu().numpy()
# y_hat   =   pred_val[0].cpu().detach().numpy()
# err = y - y_hat
# %%
# colors = np.random.rand(len(err))
# plt.scatter(y, y_hat,  c=err, alpha=0.5)
# plt.scatter(y, y_hat, s=err*0.001, c=err, alpha=0.5)
# plt.scatter(y, y_hat)
# plt.show()
