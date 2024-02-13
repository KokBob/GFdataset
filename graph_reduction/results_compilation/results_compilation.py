# -*- coding: utf-8 -*-
"""
https://realpython.com/python-zip-function/
"""
import sys
import os 
import pandas as pd
sys.path.append("..") 
from load_dataset import load_dataset 
from modelling.modelsStore import GCN0
from modelling.modelsStore import SAGE0
from modelling.preprocessing import *
import modelling.experimentationing as exps
import torch
import matplotlib.pyplot as plt
import glob as glob
# lambda zip arguments iter over them and compile(source, filename, mode)
# path_graph='./datasets/*.adjlist'
path_ds='../datasets/b2/*.adjlist'
# path_ds='../datasets/b2/*.adjlist'
list_of_files = glob.glob(f'{path_ds}')
# %%
# tutut = [f'../datasets/b3/*{x}*.adjlist/' for x in de]
# tutut = [f'../datasets/fs/*{x}*.adjlist/' for x in de]
ids = dict.fromkeys(['b2', 'b3', 'fs', 'pl'], None)
c1, c2  = 'b2', 'Bench'
def validation_loop( methodID_string, 
                    experiment_IDs_collection, 
                    ModelSet, 
                    path_to_results = './' ):
    for experiment_number in experiment_IDs_collection:
        exp_ = exps.Experiment_Graph(gfds_name, methodID_string,
                                     experiment_number, graphs, pathRes )
        try: os.mkdir(f'{pathRes}pics/')
        except: pass
        pathModel = pathRes + exp_.experiment_name + '.pt'
        pathIMG =  pathRes + 'pics/' + exp_.experiment_name + '_val.png'
        exp_.training_preparation(ModelSet)
        exp_.model.load_state_dict(torch.load(pathModel))
        exp_.model.eval()
        exp_.validate()
        exp_.validate_plot()
        plt.savefig(pathIMG)
        plt.close()

# b2_res_Bench = glob.glob(f'{path_ds}')
def wrapper_pathes_pt(_dataset_id, _experiment_id):
    path2dsres = f'../ffnet/{_dataset_id}/{_experiment_id}/*.pt'
    return glob.glob(f'{path2dsres}')
# %%
from dgl.dataloading import GraphDataLoader
lb2 = wrapper_pathes_pt('b2', 'Bench')
# %%
adj_='../datasets/b2/Beam2D.adjlist'
gfds_ = load_dataset.Beam2D(path_graph=adj_)
split_number = int(len(graphs)*.7)
graphs = graphs_preparation(D, G, X0, y)
graphs = gfds_.G
train_loader = GraphDataLoader(graphs[:split_number],shuffle=True, )
test_loader  = GraphDataLoader(graphs[split_number:],shuffle=False, ) 
validation_loop('SAGE_RF2', experiments_IDs_0, SAGE0, path_to_results = lb2[0])
# %%
dict_experiments = {'Bench': None, 
                    'ShP': None, 
                    'TrS':None, 
                    'L1':None, 
                    'WL1':None, 
                    'WL2':None}
de = list(dict_experiments.keys())
# tutut = [f'../datasets/b2/*{x}*.adjlist/' for x in de]
# dict_experiments['Bench'] = 

# intfDict = {k: intf2pptx(v) for k, v in dctPdo.items()}
# {k: v.prso() for k, v in intfDict.items()}
# %%
df = pd.DataFrame()
# %%
class rs(object):
    def __init__(self):
        self.ids = ['b2', 'b3', 'fs', 'pl']
        self.dict_experiments = {'Bench': None, 
                                 'ShP': None, 
                                 'TrS':None, 
                                 'L':None, 
                                 'WL1':None, 
                                 'WL2':None}
    def getter(self,):
        pass
f = rs()
# %%
arg_ = ['']
# GFDS_list = [load_dataset.Beam2D(path_graph=arg), 
#              load_dataset.Beam3D(path_graph=arg), 
#              load_dataset.Fibonacci(path_graph=arg), 
#              load_dataset.Plane(path_graph=arg) ]

# %%
def summary_graph(graph_):
    ln = len(graph_.nodes)
    le = len(graph_.edges)
    print(f'nodes: {ln}, edges: {le}')
def summary_edges(adj_path ,graph_):
    le = len(graph_.edges)
    print(f'{adj_path} edges: {le}')
# for gfds_ in GFDS_list: summary_graph(gfds_.G)
# %%
# for adj_, number in zip(letters, numbers):
# for adj_ in tutut:  
path_ds='../datasets/pl/*.adjlist'
list_of_files = glob.glob(f'{path_ds}')
for adj_ in list_of_files:  
    # gfds_ = load_dataset.Plane(path_graph=adj_)
    # summary_graph(gfds_.G)
    summary_edges(adj_ ,gfds_.G)
# %%

# def wrapper1(func, *args): # with star
#     func(*args)
# for _func_  in GFDS_list:
#     gfds_ = _func_
#     summary_graph(gfds_.G)
    
    
    
# G = GFDS.G

# %%

def wrapper1(func, *args): # with star
    func(*args)

def wrapper2(func, args): # without star
    func(*args)

def func2(x, y, z):
    print( x+y+z)
    
wrapper1(func2, 1, 2, 3)
wrapper2(func2, [1, 2, 3])
