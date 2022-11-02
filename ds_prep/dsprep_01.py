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
file_CF = 'dsallCF2_01.csv'
file_RF = 'dsallRF2_01.csv'
file_S = 'dsallS_01.csv'
G_adj =  'B2.adjlist'  # # grapooh defined by adjacency list 
G_ml =  'preping/B2.graphml'  

x_CF = pd.read_csv(file_CF)
x_RF = pd.read_csv(file_RF)
y = pd.read_csv(file_S)
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
