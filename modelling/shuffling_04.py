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
# json_file_name  = '../datasets/Beam3D.json'
json_file_name  = '../datasets/Fibonacci_spring.json'
D = load_dataset.dataset(json_file_name)
dkeys =D.getAvailableKeys()
X = D.selByKey('CF.CF2').T
y = D.selByKey('S.Max. Prin').T
n=-1
XList = [0,12]
yList = [0,3]
X = X[:n,XList]
y = y[:n,:] # /y = y[:n,yList] 
# %%
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=.3)
# %%
Xtrain = th.tensor(Xtrain , dtype=th.float)
ytrain = th.tensor(ytrain, dtype=th.float)
train_ds = TensorDataset(Xtrain,ytrain) 
Xtest = th.tensor(Xtest , dtype=th.float)
ytest = th.tensor(ytest, dtype=th.float)
test_ds = TensorDataset(Xtest,ytest) 
# %%
train_dl = DataLoader(train_ds, batch_size = 20, shuffle=True)
test_dl = DataLoader(test_ds, batch_size = 20, shuffle= False)
# %%
def fit(num_epochs, model, loss_fn, opt):
    running_loss = 0.
    last_loss = 0.
    for epoch in range(num_epochs):
        running_vloss = 0.0
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
        print('Training loss: ', loss_fn(model(xb), yb))
        for i, vdata in enumerate(test_dl):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
    #     # Gather data and report
    #     running_loss += loss.item()
    #     if i % 1000 == 999:
    #         last_loss = running_loss / 1000 # loss per batch
    #         print('  batch {} loss: {}'.format(i + 1, last_loss))
    #         tb_x = epoch_index * len(training_loader) + i + 1
    #         tb_writer.add_scalar('Loss/train', last_loss, tb_x)
    #         running_loss = 0.

    return running_vloss
# %%
modelF = SimpleNet(X.shape[1],y.shape[1])
# modelF = nn.Linear(X.shape()[1],y.shape()[1])
optF = th.optim.SGD(modelF.parameters(), 1e-5)
loss_fnF = F.mse_loss
running_vloss = fit(100, modelF, loss_fnF, optF)
# %%
import matplotlib.pyplot as plt
plt.plot(running_vloss.detach)
# %%
# initialize the Trainer
# trainer = Trainer()

# test the model
# trainer.test(model, dataloaders=DataLoader(test_set))

# testPred = modelF(xb)