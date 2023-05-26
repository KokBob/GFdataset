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
experiments_IDs_0 = [42,17,23,11,18,4,5,1,6,1212] # Not scaled

GT = ground_truth.GroundTruth()
# GT.beam2D(vtu_file   = GFDS.gfds_name + '_' + methodID_string + '_01.vtu')
# %% testing of scaler 
scaler = MinMaxScaler()
X0 = pd.DataFrame(df, columns=df.columns).values
XA = pd.DataFrame(df, columns=df.columns)
X1 = XA.loc[:, (XA != 0).any(axis=0)] # 
print(X0.max())
num_epochs = 500

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
# beast = vae_model.state_dict()
# torch.save(beast,pathModel) 
# %%
vae_model.load_state_dict(torch.load(pathModel))
vae_model.eval()

# %%
Z = np.zeros([2,1])
X_hat = np.zeros(X0.shape)
i = 0
for x_sample in X0:
# x_sample = X0[2]
    x_sample_device = torch.Tensor(x_sample).to(my_device) 
    z, mu, log_std = vae.encode(x_sample_device)
    z_np = z.detach().cpu().numpy().reshape([2,1])
    x_hat, mu, log_std, mse_loss = vae(x_sample_device)
    Z = np.append(Z, z_np, axis = 1)  
    X_hat[i,:] = x_hat.detach().cpu().numpy()
    i +=1
    # break
Zf = np.delete(Z, 0,1) # Z final witn initial zeros delete 
Xnorm = np.sum(np.abs(X0)**2,axis=-1)**(1./2)
# %%
plt.figure()
plt.scatter(Zf[0], Zf[1], c = Xnorm) 
plt.gca().update(dict(title='SCATTER', 
                      xlabel='axis latent: 0', 
                      ylabel = 'axis latent: 1', 
                      # ylim=(0,10)
                      ))
# plt.
# %%
index_array = np.arange(0,len(X0[:,0]))
# ex = x_hat - x_sample_device
ex = X_hat - X0
# erx = ex.detach().cpu().numpy()
erx = ex
erx_norm = np.sum(np.abs(erx)**2,axis=-1)**(1./2)

# %%
X_hat_norm = np.sum(np.abs(X_hat)**2,axis=-1)**(1./2)
X_hat_norm = X_hat_norm/ X_hat_norm.max() 
ynorm = np.sum(np.abs(y)**2,axis=-1)**(1./2)
ynorm = ynorm / ynorm.max() 
# %%
plt.figure()
# plt.scatter(index_array, erx_norm, c = X0[:,0]) 
# plt.scatter(index_array, erx_norm, c = ynorm ) 
plt.scatter(Xnorm, erx_norm, c = ynorm ) 
# %%
plt.figure()
plt.scatter(Xnorm, X_hat_norm, c = ynorm ) 
plt.gca().update(dict(title='', 
                      xlabel='Xnorm', 
                      ylabel = 'Xhatnorm', 
                      # ylim=(0,10)
                      ))
# %%
plt.figure()
plt.scatter(Xnorm, X_hat_norm, c = erx_norm) 
# %%
plt.figure()
plt.scatter(X_hat_norm, ynorm, c = erx_norm) 
plt.gca().update(dict(title='', 
                      xlabel='Xhatnorm', 
                      ylabel = 'Ynorm', 
                      # ylim=(0,10)
                      ))
# plt.scatter(Zf[0], Zf[1], c = X0[:,0]) 

# %% applicable only on beam 2D
# plt.bar(np.arange(0,len(X0[:,0])),X0[:,0])
# plt.bar(np.arange(0,len(X0[:,0])),X0[:,11], color = 'red')
# plt.bar(np.arange(0,len(X0[:,0])),X0[:,22], color = 'green')
plt.figure()
plt.bar(np.arange(0,len(X0[:,0])),Xnorm)
plt.bar(np.arange(0,len(X0[:,0])),Xnorm)

# %%
plt.figure()
plt.scatter(Zf[0], Zf[1], c = ynorm) 
plt.figure()
plt.scatter(Xnorm, ynorm, c = ynorm ) 
plt.figure()
plt.scatter(X_hat_norm, ynorm, c = ynorm ) 

# loadovani modelu ... 
# plot dataset in latent space
# plot graph 
# plot cone 
# interpolation between points 
# %% # synthetizing

