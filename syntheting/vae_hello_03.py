# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

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
# https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b

import torch
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from modelling.modelsStore import SimpleNet, SAGE0
from modelling.modelsStore import ds_splitting
from modelling.syntheting import VAE1
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

num_epochs = 1000






# SimpleNet(X0.shape[1],y.shape[1])
# for batch in test_loader:        
#     batch = batch.to(my_device)
#     # x       = batch.ndata[inputs].cpu().numpy()
#     # y       = batch.ndata[targets].c1pu().numpy()
#     # y_hat   = model(batch, batch.ndata[inputs]).cpu().detach().numpy()
#     # err     = y - y_hat


# %%
# vae = VAE(input_dim=X0.shape[1], hidden_dim=32, latent_dim=2)
vae = VAE(input_dim=X0.shape[1], hidden_dim=X0.shape[1], latent_dim=2)
# %%
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

# %%
train_loader, test_loader = ds_splitting(X0,y)
# model = SimpleNet(X0.shape[1],y.shape[1])
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
my_device = "cuda" if torch.cuda.is_available() else "cpu"    
vae = vae.to(my_device)  
# %% vae test encode and decode
for xb,yb in train_loader: 
    x = xb
    break
#%% Encode input into latent space
x = x.to(my_device)  
z, mu, log_std = vae.encode(x)
# %% Decode latent point into predicted output
x_hat = vae.decode(z)
# %% vae test forward pass
x_hat, mu, log_std, mse_loss = vae.forward(x)
# %% vae test loss
# vae_loss = vae.loss(x, x_hat, mu, log_std, mse_loss)
# %%
vault = []
inputs = 'x'
targets = 'y'
# t0 = time.time() 
total_loss = 0.0
total_mse_loss = 0.0      
for epoch in range(num_epochs):        
    total_loss = 0.0
    batch_count = 0    
     
    for xb,yb in train_loader:            
        optimizer.zero_grad()
        # batch = batch.to(my_device)
        # pred = vae(xb.to(my_device))
        
        # Forward pass
        # x = batch
        x = xb.to(my_device)  
        x_hat, mu, log_std, mse_loss = vae(x)
        loss = vae.loss(x, x_hat, mu, log_std, mse_loss_weight=0.1) # Set MSE loss weight to 0.1
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Update loss metrics
        total_loss += loss.item()
        total_mse_loss += mse_loss.item()
        
    # avg_loss = total_loss / len(dataset)
    # avg_mse_loss = total_mse_loss / len(dataset)
    avg_loss = total_loss / len(X0)
    avg_mse_loss = total_mse_loss / len(X0)
    vault.append([avg_loss, avg_mse_loss])
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, MSE Loss = {avg_mse_loss:.4f}")
# %%
import pandas as pd
df = pd.DataFrame(vault)
# %%
# df[0].plot()
df[0].plot(logy  = True)
# In this example, we first define a synthetic dataset consisting of 1000 samples of 10-dimensional Gaussian noise. We then create a PyTorch Dataset and DataLoader to load batches of data during training.
# Next, we create an instance of the VAE class with an input dimension of 10, hidden dimension of 32, and latent dimension of 2. We also create an Adam optimizer to optimize the VAE's parameters.
# We then train the VAE for 50 epochs, iterating over the batches in the dataloader and performing a forward pass, backward pass, and parameter update for each batch. During each epoch, we keep track of the total loss and MSE loss and print out the average losses at the end of the epoch.
# %%

num_samples = 10

with torch.no_grad():
    # Generate random samples from latent space
    z_samples = torch.randn(num_samples, vae.latent_dim)
    # Decode samples
    x_hat_samples = vae.decode(z_samples)
    
    # Print original and reconstructed samples
    # print("Original Samples:")
    # print(dataset[:num_samples])
    # print("Reconstructed Samples:")
    # print(x_hat_samples)


