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
df = pd.DataFrame(X0)
dfy = pd.DataFrame(y)

graphs = graphs_preparation(D, G, X0, y)
experiments_IDs = [42]
num_epochs      = 10
# %% training loop pro gcn
for experiment_number in experiments_IDs:
    exp_ = exps.Experiment_Graph(gfds_name, methodID, experiment_number, graphs)
    exp_.training_preparation(MODEL)
    exp_.training_run(num_epochs)
    exp_.validate()
