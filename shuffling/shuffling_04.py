import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append("..") # Adds higher directory to python modules path.
from load_dataset import load_dataset 
from modelling.modelsStore import SimpleNet
import torch as th
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader # co je dataloade2?
import torch.nn.functional as F
json_file_name  = '../datasets/Beam3D.json'
D = load_dataset.dataset(json_file_name)
dkeys =D.getAvailableKeys()
X = D.selByKey('CF.CF2').T
y = D.selByKey('S.Max. Prin').T

n=-1
XList = [0,12]
yList = [0,3]
X = X[:n,XList]
y = y[:n,:] # /y = y[:n,yList] 
# #
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=.3)
train_ds = TensorDataset(Xtrain,ytrain) 
test_ds = TensorDataset(Xtest,ytrain) 
# 
train_dl = DataLoader(train_ds, batch_size = 20, shuffle=True)
test_dl = DataLoader(test_ds, batch_size = 20, shuffle=True)
# %%
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
        print('Training loss: ', loss_fn(model(xb), yb))
# %%
# modelF = SimpleNet(Xt.size()[1],yt.size()[1])
modelF = nn.Linear(X.size()[1],y.size()[1])
optF = th.optim.SGD(modelF.parameters(), 1e-5)
loss_fnF = F.mse_loss
fit(100, modelF, loss_fnF, optF)
