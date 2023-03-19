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
# json_file_name  = '../datasets/b3/Beam3D.json'
# json_file_name  = '../datasets/fs/Fibonacci_spring.json'
json_file_name  = '../datasets/Plane.json'
D = load_dataset.dataset(json_file_name)
dkeys =D.getAvailableKeys()
X = D.selByKey('U.U1').T 
y = D.selByKey('S.Max. Prin').T 
n=-1

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df = pd.DataFrame(X)
X = pd.DataFrame(scaler.fit_transform(df), columns=df.columns).values
print(X.max())
# %%
def ds_splitting(X,y):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=.3)
    
    Xtrain = th.tensor(Xtrain , dtype=th.float)
    ytrain = th.tensor(ytrain, dtype=th.float)
    
    Xtest = th.tensor(Xtest , dtype=th.float)
    ytest = th.tensor(ytest, dtype=th.float)
    
    train_ds = TensorDataset(Xtrain,ytrain) 
    test_ds = TensorDataset(Xtest,ytest) 
    
    train_dl = DataLoader(train_ds, batch_size = 10, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size = 10, shuffle= True)
    return train_dl, test_dl
# %%
train_dl, test_dl = ds_splitting(X,y)
def fit(num_epochs, model, loss_fn, opt):
    running_loss = 0.
    last_loss = 0.
    vloss_ = []
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
            vl = float(running_vloss.detach())
            vloss_.append(vl)
        # Gather data and report
        if epoch % 5 == 1:
            # print(f"loss at epoch {epoch} = {mean_loss}")    # get test accuracy score
            num_correct = 0.
            num_total = 0.
            model.eval()    
            running_loss += loss.item()
        # if i % 1000 == 999:
            # last_loss = running_loss / 1000 # loss per batch
            # print('  batch {} loss: {}'.format(i + 1, last_loss))
            # tb_x = epoch_index * len(training_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            # running_loss = 0.
    # return running_vloss
    return vloss_
# %%
modelF = SimpleNet(X.shape[1],y.shape[1])
# modelF = nn.Linear(X.shape[1],y.shape[1])
# optF = th.optim.SGD(modelF.parameters(), 1e-3) # 1e-3 Smax 
optF = th.optim.Adam(modelF.parameters(), lr=1e-3)
loss_fnF = F.mse_loss
running_vloss = fit(1000, modelF, loss_fnF, optF)
# %%
import matplotlib.pyplot as plt
plt.plot(running_vloss)
plt.yscale("log")
# %%
# initialize the Trainer
# trainer = Trainer()
# test the model
# trainer.test(model, dataloaders=DataLoader(test_set))
# testPred = modelF(xb)