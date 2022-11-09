# -*- coding: utf-8 -*-
"""

Potreba data skejlovat min max X .. overim si to na prvni siti 

https://pytorch.org/docs/stable/nn.functional.html
# https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
# https://doc.dgl.ai/guide/training-node.htm
# https://medium.com/@aungkyawmyint_26195/multi-layer-perceptron-mnist-pytorch-463f795b897a
https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb

"""
# %% LIBS
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
# %% PATH SETTING
file_CF  = '../rescomp/dsallCF2_01.csv'
file_RF  = '../rescomp/dsallRF2_01.csv'
file_S   = '../rescomp/dsallS_01.csv'
G_adj    = '../preping/B2.adjlist'  # # grapooh defined by adjacency list 
G_ml     = '../preping/B2.graphml'  
# %% reading data
y = pd.read_csv(file_S).T
y = y.drop(index='Unnamed: 0')
x_CF = pd.read_csv(file_CF)
x_RF = pd.read_csv(file_RF)
x_CF = x_CF.T 
x_RF = x_RF.T
d= pd.concat([x_CF,x_RF], axis = 1)
X = d.iloc[1:,[2,8,9]] 
G = nx.read_graphml(G_ml)
seeding_magic_number = 42  
# %% Splitting and shuffling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seeding_magic_number)
X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values
# %% definice site 
in_num = X.shape[1]   
out_num = y.shape[1]

# %% # prepare data loaders
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_index, valid_index = indices[split:], indices[:split]# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(valid_index)

train_loader = th.utils.data.DataLoader(X_train, batch_size = batch_size, 
                                           # sampler = train_sampler, num_workers = num_workers)
                                             num_workers = num_workers)
# %% model architektura

import torch.nn as nn
import torch.nn.functional as F# define NN architecture




#%% arch0
# class Net(nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         # number of hidden nodes in each layer (512)
#         hidden_1 = 5
#         hidden_2 = 5
#         # linear layer (784 -> hidden_1)
#         self.fc1 = nn.Linear(1*3, 5)
#         # linear layer (n_hidden -> hidden_2)
#         self.fc2 = nn.Linear(5,5)
#         # linear layer (n_hidden -> 10)
#         self.fc3 = nn.Linear(5,5)
#         # dropout layer (p=0.2)
#         # dropout prevents overfitting of data
#         self.droput = nn.Dropout(0.2)
        
#     def forward(self,x):
#         # flatten image input
#         x = x.view(-1,1*5)
#         # add hidden layer, with relu activation function
#         x = F.relu(self.fc1(x))
#         # add dropout layer
#         x = self.droput(x)
#          # add hidden layer, with relu activation function
#         x = F.relu(self.fc2(x))
#         # add dropout layer
#         x = self.droput(x)
#         # add output layer
#         x = self.fc3(x)
#         return x# initialize the NN
model = Net()
print(model)
# %% critery and optimizer
# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()# specify optimizer (stochastic gradient descent) and learning rate = 0.01
optimizer = th.optim.SGD(model.parameters(),lr = 0.01)


# %% Loop

# number of epochs to train the model
n_epochs = 50# initialize tracker for minimum validation loss
valid_loss_min = np.Inf  # set initial "min" to infinity
for epoch in range(n_epochs):
    # monitor losses
    train_loss = 0
    valid_loss = 0
    
     
    ###################
    # train the model #
    ###################
    model.train() # prep model for training
    for data,label in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output,label)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * data.size(0)
        
        
     ######################    
    # validate the model #
    ######################
    model.eval()  # prep model for evaluation
    for data,label in valid_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output,label)
        # update running validation loss 
        valid_loss = loss.item() * data.size(0)
    
    # print training/validation statistics 
    # calculate average loss over an epoch
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, 
        train_loss,
        valid_loss
        ))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = valid_loss
