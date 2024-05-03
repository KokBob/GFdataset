# https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html

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
# from syntheting import Xhat_Generate, eval_VAE
import random
import glob 
import imageio
# %%
# GFDS = load_dataset.Beam2D()
# GFDS = load_dataset.Beam3D()
GFDS = load_dataset.Fibonacci()
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
def plot_(x_npt,y_npt,
          x_npv,y_npv,
          GFDS, 
          exp_id):
    pass
    
# %%
exp_id = experiments_IDs_0[0]
random.seed(exp_id)
X0 = pd.DataFrame(df, columns=df.columns).values
y = GFDS.y
train_loader, test_loader = ds_splitting(X0,y)

x_nb, y_nb = [], []
i = 0
for xb,yb in train_loader: 
# for xb,yb in test_loader: 
    x = xb.cpu().numpy()
    y = yb.cpu().numpy()
    x_nb.append(x)
    y_nb.append(y)
np_xb = np.concatenate(x_nb)
np_yb = np.concatenate(y_nb)

# np_xb = np_xb[~np.all(np_xb == 0, axis=1)]

# %%
# for id_ in experiments_IDs_0:
def plottping(exp_id):
    random.seed(exp_id)
    X0 = pd.DataFrame(df, columns=df.columns).values
    y = GFDS.y
    train_loader, test_loader = ds_splitting(X0,y)
    
    
    x_nb, y_nb = [], []
    for xb,yb in train_loader: 
        x = xb.cpu().numpy()
        y = yb.cpu().numpy()
        x_nb.append(x)
        y_nb.append(y)
    np_xb = np.concatenate(x_nb)
    
    # np_xb = np_xb[~np.all(np_xb == 0, axis=1)]
    
    np_yb = np.concatenate(y_nb)
    
    
    
    x_nv, y_nv = [], []    
    for xv,yv in test_loader: 
        x = xv.cpu().numpy()
        y = yv.cpu().numpy()
        x_nv.append(x)
        y_nv.append(y)
    
        
    np_xv = np.concatenate(x_nv)
    # np_xv = np_xv[~np.all(np_xv == 0, axis=1)]
    np_yv = np.concatenate(y_nv)
    
    
    
    # fig, axs = plt.subplots(2, 2, sharey=True)
    fig, axs = plt.subplots(2, 2, sharey=False)
    
    # axs[0, 0].imshow(np_xb.T)
    # axs[0, 1].imshow(np_yb.T)
    # axs[1, 0].imshow(np_xv.T)
    # axs[1, 1].imshow(np_yv.T)
    
    # axs[0, 0].imshow(np_xb.T,aspect="equal")
    # axs[0, 1].imshow(np_yb.T,aspect="equal")
    # axs[1, 0].imshow(np_xv.T,aspect="equal")
    # axs[1, 1].imshow(np_yv.T,aspect="equal")
    
    axs[0, 0].imshow(np_xb.T,aspect="auto")
    axs[0, 1].imshow(np_yb.T,aspect="auto")
    axs[1, 0].imshow(np_xv.T,aspect="auto")
    axs[1, 1].imshow(np_yv.T,aspect="auto")
    
    axs[0, 0].set_title('$\mathcal{X}_{train}$')
    axs[0, 1].set_title('$\mathcal{y}_{train}$')
    axs[1, 0].set_title('$\mathcal{X}_{test}$')
    axs[1, 1].set_title('$\mathcal{y}_{test}$')
    
    axs[0, 0].set_ylabel('node label')
    axs[1, 0].set_ylabel('node label')
    
    axs[0, 0].set_xlabel('size ')
    axs[0, 1].set_xlabel('size ')
    axs[1, 0].set_xlabel('size ')
    axs[1, 1].set_xlabel('size ')
    
    
    
    axs[0, 0].set_xticks([0, np_xb.shape[0]])
    axs[0, 1].set_xticks([0, np_yb.shape[0]])
    axs[1, 0].set_xticks([0, np_xv.shape[0]])
    axs[1, 1].set_xticks([0, np_yv.shape[0]])
    
    axs[0, 0].set_yticks([0, np_xb.shape[1]])
    axs[0, 1].set_yticks([0, np_yb.shape[1]])
    axs[1, 0].set_yticks([0, np_xv.shape[1]])
    axs[1, 1].set_yticks([0, np_yv.shape[1]])
    
    
    fig.suptitle(f'Data insight: {GFDS.gfds_name} (experiment {exp_id})')
    plt.tight_layout()
    plt.savefig(f'pics/data_insight_{GFDS.gfds_name}_{exp_id}.png')
for xpe_id in experiments_IDs_0:
# xpe_id = experiments_IDs_0[0]
    plottping(xpe_id)
plt.close('all')
# %%
import imageio
gif_name = f'pics/data_insight_{GFDS.gfds_name}.gif'
fish4     = f'pics/data_insight_{GFDS.gfds_name}*.png'
filenames = glob.glob(fish4, recursive=True)
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
# imageio.mimsave(gif_name, images)
imageio.mimsave(gif_name, images, loop=4, duration = 0.03)
