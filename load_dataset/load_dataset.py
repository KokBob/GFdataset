# -*- coding: utf-8 -*-

import pandas as pd
import json
import numpy as np
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
 
# %%
json_file_name  = '../datasets/Beam3D.json'
D = dataset(json_file_name)
dkeys =D.getAvailableKeys()
X = D.selByKey('CF.CF2')
y = D.selByKey('S.Max. Prin')

