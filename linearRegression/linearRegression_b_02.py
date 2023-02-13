#%% REFERENCES
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/
# https://www.geeksforgeeks.org/linear-regression-using-pytorch/
# https://pytorch.org/docs/stable/optim.html
# https://www.geeksforgeeks.org/how-to-scale-pandas-dataframe-columns/
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
# %% scaling 
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler

scaler = MinMaxScaler()
# scaler = StandardScaler()

X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
y_scaled = pd.DataFrame(scaler.fit_transform(y), columns=y.columns)
# 2f_norm = (df-df.min())/(df.max()-df.min())
# df_norm = pd.concat((df_norm, data.species), 1)

# %%
# x_th = th.tensor(X.values, dtype=torch.float)
# y_th = th.tensor(y.values, dtype=torch.float)
# %%
x_th = th.tensor(X_scaled.values, dtype=torch.float)
y_th = th.tensor(y_scaled.values, dtype=torch.float)
# %%
x_dataset = x_th.T
y_dataset = y_th.T

# %%

# x_dataset = x
# y_dataset = y

# And make a convenient variable to remember the number of input columns
n = 3


### Model definition ###

# First we define the trainable parameters A and b 
A = torch.randn((1, n), requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Then we define the prediction model
def model(x_input):
    return A.mm(x_input) + b


### Loss function definition ###

def loss(y_predicted, y_target): #
    return ((y_predicted - y_target)**2).sum() #root square 

### Training the model ###

# Setup the optimizer object, so it optimizes a and b.
optimizer = optim.Adam([A, b], lr=0.1)
# %%

lossnp = np.zeros([2000])
# %%
# Main optimization loop
for t in range(2000):
    # Set the gradients to 0.
    optimizer.zero_grad()
    # Compute the current predicted y's from x_dataset
    y_predicted = model(x_dataset)
    # See how far off the prediction is
    current_loss = loss(y_predicted, y_dataset)
    # Compute the gradient of the loss with respect to A and b.
    current_loss.backward()
    # Update A and b accordingly.
    optimizer.step()
    print(f"t = {t}, loss = {current_loss}")
    lossnp[t] = current_loss
    # print(f"t = {t}, loss = {current_loss}, A = {A.detach().numpy()}, b = {b.item()}")
    
# %%
plt.plot(lossnp)
# %%
# 	2	3	4
# I8	= th.tensor([-40.0	29.979	9.8267

y_predicted = model(x_dataset)
# %%
