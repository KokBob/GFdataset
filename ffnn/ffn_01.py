#%% REFERENCES
# https://www.kaggle.com/code/aakashns/pytorch-basics-linear-regression-from-scratch
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/
# https://www.geeksforgeeks.org/linear-regression-using-pytorch/
# https://pytorch.org/docs/stable/optim.html
#%% LIBS
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import networkx as nx
from sklearn.model_selection import train_test_split
import torch 
import torch as th
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
# %% PATHES
file_CF  = '../rescomp/dsallCF2_01.csv'
file_RF  = '../rescomp/dsallRF2_01.csv'
file_S   = '../rescomp/dsallS_01.csv'
G_adj    = '../preping/B2.adjlist'  # # grapooh defined by adjacency list 
G_ml     = '../preping/B2.graphml'  
# %% reading data
y = pd.read_csv(file_S).T
y = y.drop(index='Unnamed: 0')
x_CF = pd.read_csv(file_CF)
x_CF = x_CF.T 
x_RF = pd.read_csv(file_RF)
x_RF = x_RF.T
d= pd.concat([x_CF,x_RF], axis = 1)
# %%
# X = d.iloc[1:,[2,8]] #
# y = d.iloc[1:,[2,3]] #
X = d.iloc[1:,[2,8,9]] #
# y = d.iloc[1:,:] #
# yt = d.iloc[1:,:] #
# X = d.iloc[1:,[2,8,9]] #
G = nx.read_graphml(G_ml)
seeding_magic_number = 42  
# selectEach = 5
# X= X.iloc[::selectEach, :]
# y= y.iloc[::selectEach, :]
x = th.tensor(X.values, dtype=torch.float)
y = th.tensor(y.values, dtype=torch.float)
# x = th.tensor(X.values[0:100], dtype=torch.float)
# y = th.tensor(y.values[0:100], dtype=torch.float)
# %%
x_data = x
y_data = y
# %%
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype='float32')
# Targets (apples, oranges)
targets = np.array([[56, 70], [81, 101], [119, 133], [22, 37], [103, 119], 
                    [56, 70], [81, 101], [119, 133], [22, 37], [103, 119], 
                    [56, 70], [81, 101], [119, 133], [22, 37], [103, 119]], dtype='float32')
# %
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

inputs = x_data
targets =  y_data

print(inputs.size())
print(targets.size())
# %% Define dataset
train_ds = TensorDataset(inputs, targets)
train_ds[0:3]
# %%
# Define data loader
batch_size = 10
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
next(iter(train_dl))
# Define a utility function to train the model
def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
        print('Training loss: ', loss_fn(model(inputs), targets))
# %% # Define model
class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 150)
        self.act1 = nn.ReLU() # Activation function
        # self.act2 = nn.Sigmoid() # Activation function
        # self.linear2 = nn.Linear(150, 2)
        self.linear2 = nn.Linear(150, 5)
    
    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        # x = self.act2(x)
        x = self.linear2(x)
        return x
# %%
modelF = SimpleNet()
optF = torch.optim.SGD(modelF.parameters(), 1e-7)
loss_fnF = F.mse_loss
# %
fit(100, modelF, loss_fnF, optF)
# %% Generate predictions
preds = modelF(inputs)
# print(preds)
# print(targets)
# ----------- 5. check results ------------------------ #
# logits = preds
# pred = torch.argmax(logits, axis=1)
# print('Accuracy', (pred == targets.T).sum().item() / len(pred))
# %%
e = preds - targets
np_e = e.detach().numpy()
np_preds = preds.detach().numpy()
np_targets = targets.detach().numpy()
