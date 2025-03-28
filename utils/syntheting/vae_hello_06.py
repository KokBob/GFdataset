# https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b

import torch
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from modelling.modelsStore import GCN0
from modelling.modelsStore import SAGE0
from modelling.modelsStore import LR0
from modelling.modelsStore import SimpleNet
from modelling.modelsStore import ds_splitting
from modelling.syntheting import VAE
from modelling.preprocessing import * # graphs_preparation
import modelling.experimentationing as exps
from load_dataset import load_dataset 

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from evaluation import evaluate_model
from evaluation import ground_truth
import pandas as pd
from syntheting import Xhat_Generate, eval_VAE


GFDS = load_dataset.Beam2D()
# GFDS = load_dataset.Beam3D()
# GFDS = load_dataset.Fibonacci()
# GFDS = load_dataset.Plane()

MODEL, methodID_string = LR0, 'LR_RF2'
# MODEL, methodID_string = SimpleNet, 'FN_RF2'
# MODEL, methodID_string = GCN0, 'GCN_RF2'
# MODEL, methodID_string = SAGE0, 'SAGE_RF2'

gfds_name   = GFDS.gfds_name
pathRes     = GFDS.pathRes
D           = GFDS.D
X0 = GFDS.X0
y = GFDS.y
G = GFDS.G
df = pd.DataFrame(X0)
dfy = pd.DataFrame(y)

graphs = graphs_preparation(D, G, X0, y)
experiments_IDs_0 = [17] # Not scaled

# GT = ground_truth.GroundTruth()
# GT.beam2D(vtu_file   = GFDS.gfds_name + '_' + methodID_string + '_01.vtu')
# %% testing of scaler 
scaler = MinMaxScaler()
X0 = pd.DataFrame(df, columns=df.columns).values
XA = pd.DataFrame(df, columns=df.columns)
X1 = XA.loc[:, (XA != 0).any(axis=0)] # 
print(X0.max())
num_epochs = 100

vae = VAE(input_dim=X0.shape[1], hidden_dim=X0.shape[1], latent_dim=2)
# %%
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
train_loader, test_loader = ds_splitting(X0,y)
my_device = "cuda" if torch.cuda.is_available() else "cpu"    
vae = vae.to(my_device)  
# %% vae test encode and decode
for xb,yb in train_loader: 
    x = xb
    break
#%% Encode input into latent space
x = x.to(my_device)  
z, mu, log_std = vae.encode(x)
x_hat = vae.decode(z) # %% Decode latent point into predicted output

vae_model = vae
pathModel =  f'./models/{gfds_name}_VAE_epochs_{num_epochs}.pt'
vae_model.load_state_dict(torch.load(pathModel))
vae_model.eval()
X_hat, Z = Xhat_Generate(X0,vae_model, my_device)
# %%

SV = eval_VAE(vae_model, my_device )
# X_hat, Z = SV.Xhat_generate( X0[1:,:] )
X_hat, Z = SV.Xhat_generate( X0 ) # first point is outlayer
# e1 = SV.aggregate(SV.get_error())
SV.Xhat_generate(GFDS.X0)
SV.involve_targets(GFDS.y)
SV.agg_intereseting()
# app running 
# norm to max and min not to 0/1
SV.build_df_collective()
df_check = SV.df
app = SV.app_get()
# pipip
# app = GT.ViewHTML()
# app = GT.ViewHTML_error_only()
if __name__ == "__main__":    app.run_server(debug=True, port=8050)
# exit()
# %% 
# dfmi = pd.DataFrame()
