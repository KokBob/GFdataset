# %%
import torch
import torch_geometric as pyg
from load_dataset import load_dataset
from importlib import reload
from modelling import modelsStore

# %%      
# Define model
GFDS = load_dataset.Beam2D()
# %%
gfds_name = GFDS.gfds_name
pathRes  = GFDS.pathRes
# %%
from modelling.modelsStore import GCN0
import modelling.experimentationing as exps
from modelling.preprocessing import *
import pandas as pd
methodID = 'GCN2' # FFNN,SAGE0,GCN0,
MODEL = GCN0
D = GFDS.D
X0 = GFDS.X0
y = GFDS.y
G = GFDS.G
df = pd.DataFrame(X0)
dfy = pd.DataFrame(y)
# %% method refactoring 
# import dgl
# from dgl import from_networkx
from torch_geometric.utils.convert import from_networkx
# graphs = graphs_preparation(D, G, X0, y)
# D = dataset_class
# G = initial_graph
# X0 = input_features
# y = output_features
i = 0
graphs = []
for _i in D.Dataset:
    # try:
    # d_i = D.Dataset[_i]
    frame = _i.split('\\')[1]
    g_ = from_networkx(G)
    g_.name = frame
    g_.data['x'] 		= th.tensor(X0[i,:],dtype = th.float)
    g_.data['y'] 		= th.tensor(y[i,:],dtype = th.float)
    # dgl style
    # g_.ndata['x'] 		= th.tensor(X0[i,:],dtype = th.float)
    # g_.ndata['y'] 		= th.tensor(y[i,:],dtype = th.float)
    graphs.append(g_)
    i +=1
    break
    # except: pass
# %%
# experiments_IDs_0 = [42,17,]
# experiments_IDs_0 = [42,17,23,11,18,4,5,1
