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
import torch
from torch.autograd import Variable
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
# X = d.iloc[1:,[2,8]] #
# y = d.iloc[1:,[2,3]] #
X = d.iloc[1:,[2,8,9]] #
y = d.iloc[1:,:] #
# X = d.iloc[1:,[2,8,9]] #
G = nx.read_graphml(G_ml)
seeding_magic_number = 42  
selectEach = 5
# X= X.iloc[::selectEach, :]
# y= y.iloc[::selectEach, :]
# x = th.tensor(X.values, dtype=torch.float)
# y = th.tensor(y.values, dtype=torch.float)
x = th.tensor(X.values[0:100], dtype=torch.float)
y = th.tensor(y.values[0:100], dtype=torch.float)
# %%


x_dataset = x.T
y_dataset = y.T
# x_dataset = x
# y_dataset = y

# And make a convenient variable to remember the number of input columns
n = 3


### Model definition ###



# x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
# y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
# x_data = x.T
# y_data = y.T
x_data = x
y_data = y

# -*- coding: utf-8 -*-
# https://www.kaggle.com/code/aakashns/pytorch-basics-linear-regression-from-scratch/notebook
# Import Numpy & PyTorch
import numpy as np
import torch

#%% Create tensors.
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)
#%% Print tensors
print(x)
print(w)
print(b)
# 
#%% Arithmetic operations
y = w * x + b
print(y)
#%% Compute gradients
y.backward()
# Display gradients
print('dy/dw:', w.grad)
print('dy/db:', b.grad)
# %%
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')
# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')
# %
# yeild_apple  = w11 * temp + w12 * rainfall + w13 * humidity + b1
# yeild_orange = w21 * temp + w22 * rainfall + w23 * humidity + b2
# %
# Convert inputs and targets to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

inputs = x_data
targets = y_data

print(inputs.size())
print(targets.size())
# %
# Weights and biases
targets.shape
# torch.Size([5, 2])
w = torch.randn(targets.shape[1], inputs.shape[1], requires_grad=True)
# w = torch.randn(2, 3, requires_grad=True)

# b = torch.randn(2, requires_grad=True)
b = torch.randn(targets.shape[1], requires_grad=True)
print(w.size())
print(b.size())
#
# %% Define the model
def model(x):
    return x @ w.t() + b
# 
# Generate predictions
preds = model(inputs)
print(preds)
# %
# Compare with targets
print(targets)
# %%
# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()
# Compute loss
loss = mse(preds, targets)
print(loss)
# %%
# Compute gradients
loss.backward()
# %%
# Gradients for weights
print(w)
print(w.grad)
# %%
# Gradients for bias
print(b)
print(b.grad)
# %% Finally, we'll reset the gradients to zero before moving forward, because PyTorch accumulates gradients.
w.grad.zero_()
b.grad.zero_()
print(w.grad)
print(b.grad)
print(w)
# Generate predictions
preds = model(inputs)
print(preds)
# %% Calculate the loss
loss = mse(preds, targets)
print(loss)
# %% Compute gradients
loss.backward()
#%% Adjust weights & reset gradients
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()
print(w)
# %%# Calculate loss
# Calculate loss
# With the new weights and biases, the model should have a lower loss.
preds = model(inputs)
loss = mse(preds, targets)
print(loss)
# %%
# Train for 100 epochs
for i in range(1000):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()
#%% Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print(loss)
# %%
# Print predictions
print(preds)
# Compare with targets
print(targets)




