# -*- coding: utf-8 -*-
"""
update Nov2nd22
load X .. CF and RF 
load G 
add to G values CF and RF 
"""
import pandas as pd
import networkx as nx
import random 
file_CF = 'C:/CAE/dummies/gnfe/physgnn/rescomp/dsallCF2_01.csv'
file_RF = 'C:/CAE/dummies/gnfe/physgnn/rescomp/dsallRF2_01.csv'
file_S = 'C:/CAE/dummies/gnfe/physgnn/rescomp/dsallS_01.csv'
G_adj =  'C:/CAE/dummies/gnfe/physgnn/preping/B2.adjlist'  # # grapooh defined by adjacency list 
G_ml =  'C:/CAE/dummies/gnfe/physgnn/preping/B2.graphml'  
# %% reading data


y = pd.read_csv(file_S)
# %%
x_CF = pd.read_csv(file_CF)
x_CF = x_CF.T 
x_CF = x_CF.iloc[1:,2]
# %%
x_RF = pd.read_csv(file_RF)
x_RF = x_RF.T
x_RF = x_RF.iloc[1:,[3,4]]
# %%
X = pd.DataFrame([x_RF.values,x_CF.values])

# %% reading graph via ml 
# G = nx.read_adjlist(G_adj)
# [G0] https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.set_node_attributes.html
G = nx.read_graphml(G_ml)
# %% seeding
# pro nejakou prokazatelnost pak navrhnout treba 3 loopy seedingu 
seeding_magic_number = 42  # 27, fibo
random.seed(seeding_magic_number)
# %% dataframes combing
#[0] https://datacarpentry.org/python-ecology-lesson/05-merging-data/
# dataset = pd.concat([x_CF, x_RF, y])
# %% shuffling 
#? pandas shuffle scikit learn: 
#       https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html
# x_CF_list = list(x_CF)
# x_CF_shuffled = random.shuffle(x_CF_list)
from sklearn.utils import shuffle
x_CF_shuffled = shuffle(x_CF.T, random_state=seeding_magic_number).T
# dve moznosti 
# a) muzu z tohoto zamichani pouzit columns ids a to aplikovat na y a dalsi dataframe 
x_RF_shuffled  = x_RF[x_CF_shuffled.columns]
y_shuffled= y[x_CF_shuffled.columns]

# %% multicolumns 
# b) vytvorit multidimensionalni dataset pres multicolumns, 
#   [1] https://datacarpentry.org/python-ecology-lesson/05-merging-data/
#   [2] https://stackoverflow.com/questions/36760414/how-to-create-pandas-dataframes-with-more-than-2-dimensions
#   [3] https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html
#   [4] https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html#cookbook-multi-index


# tohle nefunguje ...
# data = pd.DataFrame(x_CF,3, index = 'x_CF')
# data = pd.DataFrame(x_RF,3, index = 'x_RF')
# data = pd.DataFrame(y,3, index = 'y')
# data = pd.DataFrame(x_CF,3, )
# data = pd.DataFrame(x_RF,3)
# data = pd.DataFrame(y,3, )
# %% grafovy dataset ... 
# c) vytvorit graf a postupovat standartne jak se to s grafama dela 
# %% Splitting and shuffling
# from sklearn.model_selection import train_test_split
#   [5] https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# X = x_RF.T
# y = y.T
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seeding_magic_number)
# %%
# X = pd.DataFrame([x_CF.loc[2].values,x_RF.loc[3::].values])
# X_train, X_test = 
# DataFrame.drop()
# [5] https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
# import os
# import torch
# from torch import nn
# # from torch.utils.data import DataLoader
# # from torchvision import datasets, transforms

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(1*3, 50),
#             nn.ReLU(),
#             nn.Linear(50, 50),
#             nn.ReLU(),
#             nn.Linear(50, 5),
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
    
    
    
