# -*- coding: utf-8 -*-
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append("..") 
from load_dataset import load_dataset 
from modelling.modelsStore import GCN0
from modelling.modelsStore import SAGE0
from modelling.preprocessing import *
import modelling.experimentationing as exps
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
import torch
import time
from dgl.dataloading import GraphDataLoader

GFDS = load_dataset.Beam3D()
gfds_name = GFDS.gfds_name
pathRes  = GFDS.pathRes
methodID = 'GCN_RF2' 
MODEL = GCN0
D = GFDS.D
X0 = GFDS.X0
y = GFDS.y
G = GFDS.G
print(X0.max())
scaler = MinMaxScaler()
df = pd.DataFrame(X0)
dfy = pd.DataFrame(y)
# X0 = pd.DataFrame(scaler.fit_transform(df), columns=df.columns).values
print(y.max())
# y  = pd.DataFrame(scaler.fit_transform(dfy), columns=dfy.columns).values
print(X0.max())
print(y.max())

graphs = graphs_preparation(D, G, X0, y)

experiments_IDs = [42]
# loss_fn         = F.mse_loss
num_epochs      = 10
# %% training loop pro gcn
for experiment_number in experiments_IDs:
    exp_ = exps.Experiment_Graph(gfds_name, methodID, experiment_number, graphs)
    exp_.training_preparation(MODEL)
    exp_.training_run(num_epochs)
    exp_.validate()
    # experiment_evaluation(experiment_name,
    #                       pathRes,
    #                       model,
    #                       epochs,
    #                       time_elapsed,
    #                       losses, 
    #                       losses_val)
# %% validate
# model       = exp_.model
# inputs      = exp_.inputs
# targets     = exp_.targets
# # for batch in exp_.test_loader:   break
# for batch in exp_.test_loader:   
#     batch = batch.to(exp_.my_device)
#     pred_val = model(batch, batch.ndata[exp_.inputs])
    
#     x       = batch.ndata[inputs].cpu().numpy()
#     y       = batch.ndata[targets].cpu().numpy()
#     y_hat   = pred_val.cpu().detach().numpy()
#     err     = y - y_hat


#     # print(y)
#     # print(y_hat)
#     print(err.sum())
# def root_max_square_error(pred, target):
#     error = torch.abs(pred - target)
#     max_error = torch.max(error)
#     return torch.sqrt(max_error**2)


# e2 = root_max_square_error(model(batch, batch.ndata[inputs]), batch.ndata[targets])
# %%
# https://stackoverflow.com/questions/44552031/sklearnstandardscaler-can-i-inverse-the-standardscaler-for-the-model-output
# from sklearn.preprocessing import StandardScaler
# data = [[1,1], [2,3], [3,2], [1,1]]
# scaler = StandardScaler()
# scaler.fit(data)
# y = scaler.fit_transform(dfy)
# scaled = scaler.transform(dfy)
# scaled = scaler.transform(data)
# for inverse transformation
# inversed = scaler.inverse_transform(scaled)
# inversed = scaler.inverse_transform(y)

# print(y[i])
# print(inversed[i])
#         for batch in test_loader:        
#             batch = batch.to(my_device)
#             pred_val = model(batch, batch.ndata[inputs])
#             loss_val = loss_fn(pred_val, batch.ndata[targets].to(my_device))
#             total_loss_val += loss_val.detach()
#             batch_count_val += 1        
#             mean_loss_val = total_loss_val / batch_count_val
#             total_val = ((pred -batch.ndata[targets])/(pred -batch.ndata[targets]).sum()).sum()
#         if epoch % 5 == 1:
#             print(f"validation accuracy = {loss_val} , {total_val}")
#             # print(f"validation accuracy = {mean_loss_val} , {total_val}") # zatim neaplikovat
#         # losses_val.append(loss_val.detach())
#         losses_val.append(mean_loss_val)
#     pathModel = pathRes + experiment_name + '.pt'
#     torch.save(model.state_dict(),pathModel)
#     experiment_evaluation(experiment_name,
#                      pathRes,
#                      model,
#                      epochs,
#                      time_elapsed,
#                      losses, 
#                      losses_val)
