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
# %%
# yeild_apple  = w11 * temp + w12 * rainfall + w13 * humidity + b1
# yeild_orange = w21 * temp + w22 * rainfall + w23 * humidity + b2
# %%
# Convert inputs and targets to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print(inputs)
print(targets)
# %%
# Weights and biases
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
print(w)
print(b)
#
# %% Define the model
def model(x):
    return x @ w.t() + b
# %%
# Generate predictions
preds = model(inputs)
print(preds)
# %%
# Compare with targets
print(targets)
# %%
# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()
# %%
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
# %%
print(w)
# %%
# 
# Generate predictions
preds = model(inputs)
print(preds)
# %% Calculate the loss
loss = mse(preds, targets)
print(loss)
# %% Compute gradients
# 
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
for i in range(100):
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
