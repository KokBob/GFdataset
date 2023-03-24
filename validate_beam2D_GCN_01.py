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

GFDS = load_dataset.Beam2D()
gfds_name = GFDS.gfds_name
pathRes  = GFDS.pathRes
# methodID = 'GCN' # FFNN,SAGE0,GCN0,
methodID = 'GCN_RF2' 
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
experiments_IDs = [42,17,23,11,18,4,5,1,6,1212]
loss_fn = F.mse_loss
num_epochs = 500
fig, axs = plt.subplots(2, 2)
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

    pathModel = pathRes + experiment_name + '.pt'
    inputs = 'x'
    targets = 'y'

    model.load_state_dict(torch.load(pathModel))
    model.eval()

    for batch in test_loader:        
        batch = batch.to(my_device)
        x       = batch.ndata[inputs].cpu().numpy()
        y       = batch.ndata[targets].cpu().numpy()
        y_hat   = model(batch, batch.ndata[inputs]).cpu().detach().numpy()
        err     = y - y_hat
    
        axs[0, 0].scatter(x, y, c=err, alpha=0.5)
        axs[0, 1].scatter(x, y_hat, c=err, alpha=0.5)
        axs[1, 0].scatter(y_hat, err, c=err, alpha=0.5)
        axs[1, 1].scatter(y, y_hat, c=err, alpha=0.5)
        # plt.scatter(y_hat, err,  c=err, alpha=0.5)
        # plt.scatter(y, y_hat,c=err, alpha=0.5)
        # plt.scatter(x, err,c=err, alpha=0.5)
        # plt.hist(err, bins = 120) 
        # plt.show()
        break
