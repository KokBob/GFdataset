# -*- coding: utf-8 -*-

import pandas as pd
import json
import numpy as np
from dgl import from_networkx
import networkx as nx
# # mpd = pd.MultiIndex.from_frame(D)
class dataset(object):
    def __init__(self, json_file_name):
        self.Dataset = json.load(open(json_file_name ))
    def getAvailableKeys(self,):
        for d in self.Dataset:
            d_ = self.Dataset[d]
            self.d_ = d_['data']
            self.d_keys = list(self.d_[0].keys())
            break
        return self.d_keys
    def init_array(self,):
        pd0 = pd.DataFrame.from_dict(self.d_) 
        self.number_frames = len(self.Dataset)
        self.number_nodes = pd0.shape[0]
        self.initial_array = np.zeros([self.number_nodes,self.number_frames])
        return self.initial_array 
    
    def selByKey(self,key_name):
        # vracim pouze numpy data , ktery by pak mohli byt dal ulozeny do numoy napriklad pro github etc.
        self.init_array()
        variable_array = np.zeros([self.number_nodes,self.number_frames])
        i = 0
        for nameFrame in self.Dataset:
            d_ = self.Dataset[nameFrame]
            d_ = d_['data']
            pd_ = pd.DataFrame.from_dict(d_)
            variable_array[:,i] = pd_[key_name].values
            i+=1
        return variable_array
    
class Beam2D(object):
    def __init__(self, key_input = None):
         
        self.gfds_name = 'Beam2D'
        self.json_file_name  = '../datasets/b2/'+self.gfds_name +'.json'
        self.path_graph      = '../datasets/b2/'+self.gfds_name +'.adjlist'
        self.pathRes         = 'b2/'
        D = dataset(self.json_file_name )
        self.D = D
        self.dkeys =D.getAvailableKeys()
        # if not key_input:
        self.X0 = D.selByKey('RF.RF2').T 
        # else: 
            # self.X0 = D.selByKey(key_input).T 
        self.y = D.selByKey('S.Max. Prin').T 
        self.G = nx.read_adjlist(self.path_graph).to_directed() 
        self.g_ = from_networkx(self.G)
    def get_summary(self,):
        g_ = self.g_
        str_summary = f'{self.gfds_name} consists of :: \n nodes: {len(g_.nodes())}, edges: {len(g_.edges()[0])}'
        print(str_summary)
class Beam3D(object):
    def __init__(self, ):
        self.gfds_name = 'Beam3D'
        self.json_file_name  = '../datasets/b3/'+self.gfds_name +'.json'
        self.path_graph      = '../datasets/b3/'+self.gfds_name +'.adjlist'
        self.path_vtk_nds    = '../datasets/b3/'+self.gfds_name +'.vtk_nds'
        self.path_vtk_els    = '../datasets/b3/'+self.gfds_name + '.vtk_els'
        self.pathRes         = './b3/'
        D = dataset(self.json_file_name )
        self.D = D
        self.dkeys =D.getAvailableKeys()
        # self.X0 = D.selByKey('U.U1').T 
        self.X0 = D.selByKey('RF.RF2').T 
        self.y = D.selByKey('S.Max. Prin').T 
        self.G = nx.read_adjlist(self.path_graph).to_directed() 
        self.g_ = from_networkx(self.G)
    def get_summary(self,):
        g_ = self.g_
        str_summary = f'{self.gfds_name} consists of :: \n nodes: {len(g_.nodes())}, edges: {len(g_.edges()[0])}'
        print(str_summary)
    def input_nodes(self,):
        self.input_nodes_list = [0,12]
        # get label of input nodes on level of input file
class Fibonacci(object):
    def __init__(self, ):
        self.gfds_name = 'Fibonacci'
        self.json_file_name  = '../datasets/fs/Fibonacci_spring.json'
        self.path_graph     = '../datasets/fs/Fibonacci_spring.adjlist'
        # self.pathRes  = 'fs0/'
        self.pathRes  = 'fs/'
        D = dataset(self.json_file_name )
        self.D = D
        self.dkeys =D.getAvailableKeys()
        # self.X0 = D.selByKey('U.U1').T 
        self.X0 = D.selByKey('RF.RF3').T 
        self.y = D.selByKey('S.Max. Prin').T 
        self.G = nx.read_adjlist(self.path_graph).to_directed() 
        self.g_ = from_networkx(self.G)
    def get_summary(self,):
        g_ = self.g_
        str_summary = f'{self.gfds_name} consists of :: \n nodes: {len(g_.nodes())}, edges: {len(g_.edges()[0])}'
        print(str_summary)
    def input_nodes(self,):
        self.keysMakinfSense = { 
            # options , None... no sense to even bother, 0,1 ... node label
            # True.. potential all nodes 
            'CF.CF1': [0],
            'CF.CF2': None,
            'CF.CF3': None ,
            'RF.RF1': [1403,...,1425], # data.loc[~(data==0).all(axis=1)].index.tolist()
            'RF.RF2': [1403,...,1425], # data.loc[~(data==0).all(axis=1)].index.tolist()
            'RF.RF3': [1403,...,1425], # data.loc[~(data==0).all(axis=1)].index.tolist()
            'U.U1': True,
            'U.U2': True,
            'U.U3': True,
            'LE.Max. Prin': True, 
            'S.Mises': True,
            'S.Max. Prin': True ,
            }
class Plane(object):
    def __init__(self, ):
        self.gfds_name = 'Plane'
        self.json_file_name  = '../datasets/pl/Plane.json'
        self.path_graph      = '../datasets/pl/Plane.adjlist'
        self.pathRes  = 'pl/'
        D = dataset(self.json_file_name )
        self.D = D
        self.dkeys =D.getAvailableKeys()
        self.X0 = D.selByKey('U.U2').T # Plane
        self.y = D.selByKey('S.Max. Prin').T 
        self.G = nx.read_adjlist(self.path_graph).to_directed() 
        self.g_ = from_networkx(self.G)
    def get_summary(self,):
        g_ = self.g_
        str_summary = f'{self.gfds_name} consists of :: \n nodes: {len(g_.nodes())}, edges: {len(g_.edges()[0])}'
        print(str_summary)
    def input_nodes(self,):
        self.keysMakinfSense = { 
            # options , None... no sense to even bother, 0,1 ... node label
            # True.. potential all nodes 
            'CF.CF1': [0],
            'CF.CF2': None,
            'CF.CF3': None ,
            'RF.RF1': [1403,...,1425], # data.loc[~(data==0).all(axis=1)].index.tolist()
            'RF.RF2': [1403,...,1425], # data.loc[~(data==0).all(axis=1)].index.tolist()
            'RF.RF3': [1403,...,1425], # data.loc[~(data==0).all(axis=1)].index.tolist()
            'U.U1': True,
            'U.U2': True,
            'U.U3': True,
            'LE.Max. Prin': True, 
            'S.Mises': True,
            'S.Max. Prin': True ,
            }
        # self.input_nodes_list = [0,12]
        # get label of input nodes on level of input file       
        # n=5
        # X = X[[0,12],:n]
        # y = y[:,:n]
# %% Example
# json_file_name  = '../datasets/Beam3D.json'
# json_file_name  = '../datasets/Fibonacci_spring.json'
# json_file_name  = '../datasets/Plane.json'
# D = dataset(json_file_name)
# %%

# dkeys =D.getAvailableKeys()
# X = D.selByKey('CF.CF1')
# data = pd.DataFrame(X)
# data = X[~np.all(X == 0, axis=1)] # 
# y = D.selByKey('S.Max. Prin')

# %
# data = data.loc[~(data==0).all(axis=1)]
# data.loc[~(data==0).all(axis=1)]
# X.loc[(df!=0).any(axis=1)] # And for those who like symmetry, this also works...
