import sys
import pandas as pd
import numpy as np
sys.path.append("..") # Adds higher directory to python modules path.
from load_dataset import load_dataset 

import torch as th
from torch.utils.data import TensorDataset,DataLoader # co je dataloade2?

json_file_name  = '../datasets/Beam3D.json'
D = load_dataset.dataset(json_file_name)
dkeys =D.getAvailableKeys()
X = D.selByKey('CF.CF2').T
y = D.selByKey('S.Max. Prin').T
# %
n=10
XList = [0,12]
yList = [0,3]
X = X[:n,XList]
y = y[:n,yList]
C = np.concatenate([X,y], axis = 1)
# %%
pdD = pd.DataFrame(C)
pdDs = pdD.sample(frac=1)
# %%
X_ds = pdDs[[0,1]].values 
y_ds = pdDs[[2,3]].values
# %%
Xt = th.tensor(X_ds , dtype=th.float)
yt = th.tensor(y_ds, dtype=th.float)
dataset = TensorDataset(Xt,yt) 
