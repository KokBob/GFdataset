# -*- coding: utf-8 -*-
"""
Created on Sat May  6 15:27:31 2023
@author: marek
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import plotly.graph_objects as go
import dash
from dash import html 
from dash import dcc
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc  # import dash_core_components as dcc
from dash import html # import das
import plotly.express as px
import plotly.graph_objects as go

'''
refs
https://www.youtube.com/watch?v=V3pzxngeLqw
# https://pandas.pydata.org/docs/user_guide/merging.html
# https://www.edureka.co/community/51168/pandas-fillna-with-another-column
# https://plotly.com/python/marker-style/
# https://plotly.com/python/graphing-multiple-chart-types/
'''




class VAE1(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE1, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim) # Output mean and log std dev
        )
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Regression loss metric
        self.mse_loss_fn = nn.MSELoss(reduction='mean')
        
    def encode(self, x):
        # Compute mean and log std dev
        h = self.encoder(x)
        mu, log_std = torch.chunk(h, 2, dim=-1)
        
        # Sample from Gaussian distribution
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z = mu + std * eps
        
        return z, mu, log_std
    
    def decode(self, z):
        # Compute predicted output
        h = self.decoder(z)
        x_hat = torch.sigmoid(h)
        
        return x_hat
    
    def forward(self, x):
        # Encode input into latent space
        z, mu, log_std = self.encode(x)
        
        # Decode latent point into predicted output
        x_hat = self.decode(z)
        
        # Compute MSE loss
        self.mse_loss = self.mse_loss_fn(x_hat, x)
        
        return x_hat, mu, log_std, self.mse_loss
    
    def loss(self, x, x_hat, mu, log_std, mse_loss_weight=1.0):
        # Compute reconstruction loss
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        
        # Compute KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_std - mu.pow(2) - log_std.exp())
        
        # Compute total loss
        total_loss = recon_loss + kl_loss + mse_loss_weight * self.mse_loss
        
        return total_loss
    
class VAE2(nn.Module):
    def __init__(self, latent_dims):  
        super(VAE2, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)  
        self.linear1 = nn.Linear(3*3*32, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z  

    
def Xhat_Generate(X0,vae_model, my_device):
    Z = np.zeros([2,1]) # Z = np.zeros(z.shape) 
    X_hat = np.zeros(X0.shape)
    i = 0
    for x_sample in X0:
    # x_sample = X0[2]
        x_sample_device = torch.Tensor(x_sample).to(my_device) 
        z, mu, log_std = vae_model.encode(x_sample_device)
        z_np = z.detach().cpu().numpy().reshape([2,1])
        x_hat, mu, log_std, mse_loss = vae_model(x_sample_device)
        Z = np.append(Z, z_np, axis = 1)  
        X_hat[i,:] = x_hat.detach().cpu().numpy()
        i +=1
    Zf = np.delete(Z, 0,1) # Z final witn initial zeros delete 
    return X_hat, Zf

class eval_VAE():
    def __init__(self, VAE_MODEL, DEVICE):  
        self.VAE_MODEL, self.DEVICE = VAE_MODEL, DEVICE
    def involve_targets(self, TARGETS): self.targets = TARGETS
        
    def Xhat_generate(self, INPUTS):
        Z = np.zeros([2,1]) # Z = np.zeros(z.shape) 
        X_hat = np.zeros(INPUTS.shape)
        i = 0
        for x_sample in INPUTS:
        # x_sample = X0[2]
            x_sample_device = torch.Tensor(x_sample).to(self.DEVICE) 
            z, mu, log_std = self.VAE_MODEL.encode(x_sample_device)
            z_np = z.detach().cpu().numpy().reshape([2,1])
            x_hat, mu, log_std, mse_loss = self.VAE_MODEL(x_sample_device)
            Z = np.append(Z, z_np, axis = 1)  
            X_hat[i,:] = x_hat.detach().cpu().numpy()
            i +=1
        Zf = np.delete(Z, 0,1) # Z final witn initial zeros delete 
        self.X, self.X_hat = INPUTS, X_hat
        self.Zf = Zf
        
        return X_hat, Zf       
    def get_error(self):
        self.error = self.X_hat - self.X
        return self.error
    def aggregate(self, VAR):
        VAR_AGG = np.sum(np.abs(VAR)**2,axis=-1)**(1./2)
        return VAR_AGG
    def agg_intereseting(self): 
        # try: 
        self.get_error()
        L = list(map(self.aggregate, [self.X, self.X_hat, self.Zf.T, 
                                      self.targets, self.error]))
        self.X_agg      =  L[0]
        self.X_hat_agg  =  L[1]
        self.Z_agg      =  L[2]
        self.y_agg      =  L[3]
        self.e_agg      =  L[4]
        
    # def plot_(self):
    #     plt.figure()
    #     # plt.scatter(Xnorm, X_hat_norm, c = ynorm ) 
    #     # plt.scatter(Xnorm, X_hat_norm, c = ynorm ) 
    #     plt.gca().update(dict(title='', 
    #                           xlabel='Xnorm', 
    #                           ylabel = 'Xhatnorm', 
    #                           # ylim=(0,10)
    #                           ))
    def build_df_collective(self,):
        df = pd.DataFrame(data = [  self.X_agg, 
                                    self.X_hat_agg,
                                    self.Z_agg,
                                    self.y_agg,
                                    self.e_agg],)
        df = df.T
        df.columns = ['X_agg', 'X_hat_agg', 'Z_agg', 'y_agg', 'e_agg']
        self.df = df
                               # columns = ['X_agg', 'X_hat_agg', 'Z_agg', 'y_agg'])
    def app_get(self,):
        # df =  pd.DataFrame(self.X_hat_agg)
        df =  self.df
        app = dash.Dash(__name__)
        fig = px.scatter_3d(df, x = df.index, y = 'X_hat_agg', z = 'y_agg', color = 'Z_agg')
        fig1 = px.scatter_3d(df, x = 'X_agg', y = 'X_hat_agg', z = 'y_agg', color = 'Z_agg')
        fig2 = px.scatter_3d(df, x = 'e_agg', y = 'X_hat_agg', z = 'y_agg', color = 'y_agg')
        fig3 = px.scatter_3d(df, x = np.ones(df.shape[0]), y = 'X_hat_agg', z = 'y_agg', color = 'y_agg')
        
        fig3.add_trace([
            'type': 'scatter3d',
            df, 
            x = np.ones(df.shape[0])+2, 
            y = 'X_agg', 
            z = 'y_agg', 
            color = 'y_agg')
            ]
            )
        # fig3.update_traces()
        app.layout = html.Div(children=[
            html.H1(children='VIOLIN '),
            html.Div(children=''' MSE of validation batch  '''),

            dcc.Graph(id='isx', figure=fig, style={'display': 'inline-block'}),
            dcc.Graph(id='XYZ', figure=fig1, style={'display': 'inline-block'}),
            dcc.Graph(id='XYe', figure=fig2, style={'display': 'inline-block'}),
            dcc.Graph(id='iXY', figure=fig3, style={'display': 'inline-block'}),
            dcc.Graph(id='id3XY', figure=fig3, style={'display': 'inline-block'}),

        ])
        self.app = app
        return app
    
        
