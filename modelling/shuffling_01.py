import sys
import pandas as pd
import numpy as np
sys.path.append("..") # Adds higher directory to python modules path.
from load_dataset import load_dataset 
from random import shuffle
import torch as th
from torch.utils.data import Dataset, TensorDataset,DataLoader # co je dataloade2?


json_file_name  = '../datasets/Beam3D.json'
D = load_dataset.dataset(json_file_name)
dkeys =D.getAvailableKeys()
X = D.selByKey('CF.CF2').T
y = D.selByKey('S.Max. Prin').T
# %%


# %%
n=10
X = X[:n,[0,12]]
y = y[:n,[0,]]
# %% tohle nefunguje
arr = [X,y]
D = np.random.shuffle(arr)
# a = X
# a1=a[:,th.randperm(a.size()[1])] # tohle nefunguje
# %%
pdX = pd.DataFrame(X)
pdXs = pdX.sample(frac=1)
# %%
Xt = th.tensor(X, dtype=th.float)
yt = th.tensor(y, dtype=th.float)
# train_ds = TensorDataset(inputs, targets)
# dataset = Dataset(Xt,yt) 
dataset = TensorDataset(Xt,yt) 

# %%

# dataset = dataset.shuffle()
# dataLoader_full = DataLoader(dataset, )
# dataLoader_full_shuffled = DataLoader(dataset, shuffle=True)
# # %%
# for batch in dataLoader_full:
    # batch
    # %%
# for batch in dataLoader_full_shuffled:
    # batch
    # >>> Batch(x=[1024, 21], edge_index=[2, 1568], y=[512], batch=[1024])
# %%
# pd0 = pd.DataFrame(dataLoader_full.dataset)
# pd1 = pd.DataFrame(dataLoader_full_shuffled.dataset)
# l0 = list(dataLoader_full.dataset)
# l1 = list(dataLoader_full_shuffled.dataset)
# %%
# suppose dataset is the variable pointing to whole datasets
# N = len(dataset)
# import numpy
# from torch.utils.data import Subset
# generate & shuffle indices
# indices = numpy.arange(N)
# indices = numpy.random.permutation(indices)
# there are many ways to do the above two operation. (Example, using np.random.choice can be used here too

# select train/test/val, for demo I am using 70,15,15
# train_indices = indices [:int(0.7*N)]
# val_indices = indices[int(0.7*N):int(0.85*N)]
# test_indices = indices[int(0.85*N):]

# train_dataset = Subset(dataset, train_indices)
# val_dataset = Subset(dataset, val_indices)
# test_dataset = Subset(dataset, test_indices)