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
import random
import glob 
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


plt.figure()
plt.subplot(2,1,1)
plt.imshow(df.values.T)
plt.subplot(2,1,2)
plt.imshow(dfy.values.T)

# %%
# df2 = df.interpolate(method='polynomial', order=2)
# dfy2 = dfy.interpolate(method='polynomial', order=2)
# plt.figure()
# plt.subplot(2,1,1)
# plt.imshow(df2.values.T)
# plt.subplot(2,1,2)
# plt.imshow(dfy2.values.T)
# %%

# graphs = graphs_preparation(D, G, X0, y)
experiments_IDs_0 = [17] # Not scaled
X0 = pd.DataFrame(df, columns=df.columns).values
XA = pd.DataFrame(df, columns=df.columns)
X1 = XA.loc[:, (XA != 0).any(axis=0)] # 

experiments_IDs_0 = [42,17,23,11,18,4,5,1,6,1212] # Not scaled
# for id_ in experiments_IDs_0:
def plottping(exp_id):
    random.seed(exp_id)
    X0 = pd.DataFrame(df, columns=df.columns).values
    y = GFDS.y
    train_loader, test_loader = ds_splitting(X0,y)
    
    
    x_np, y_np = [], []
    i = 0
    for xb,yb in train_loader: 
    # for xb,yb in test_loader: 
        x = xb.cpu().numpy()
        y = yb.cpu().numpy()
        x_np.append(x)
        y_np.append(y)
    
    plt.figure()
    np_xb = np.concatenate(x_np)
    np_yb = np.concatenate(y_np)
    plt.subplot(2,2,1)
    plt.imshow(np_xb.T)
    plt.subplot(2,2,2)
    plt.imshow(np_yb.T)
    x_np, y_np = [], []
    for xb,yb in test_loader: 
        x = xb.cpu().numpy()
        y = yb.cpu().numpy()
        x_np.append(x)
        y_np.append(y)
    
        
    np_xb = np.concatenate(x_np)
    np_yb = np.concatenate(y_np)
    plt.subplot(2,2,3)
    plt.imshow(np_xb.T)
    plt.subplot(2,2,4)
    plt.imshow(np_yb.T)
    plt.savefig(f'data_insight_{exp_id}.png')


for xpe_id in experiments_IDs_0:
    plottping(xpe_id)
# %%

import imageio
import glob 
fish4     = f'data_insight_*.png'
filenames = glob.glob(fish4, recursive=True)
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
# imageio.mimsave('movie.gif', images)
imageio.mimsave(gif_name, fileList, loop=4, duration = 0.3)
