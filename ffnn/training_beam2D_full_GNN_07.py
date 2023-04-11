# -*- coding: utf-8 -*-
import sys
import pandas as pd
sys.path.append("..") 
from load_dataset import load_dataset 
from modelling.modelsStore import GCN0
from modelling.modelsStore import SAGE0
from modelling.preprocessing import *
import modelling.experimentationing as exps
import torch
import matplotlib.pyplot as plt


GFDS = load_dataset.Beam2D()
gfds_name   = GFDS.gfds_name
pathRes     = GFDS.pathRes
D           = GFDS.D

X0 = GFDS.X0
y = GFDS.y
G = GFDS.G
df = pd.DataFrame(X0)
dfy = pd.DataFrame(y)

graphs = graphs_preparation(D, G, X0, y)
# experiments_IDs = [42,17,]
experiments_IDs_0 = [42,17]

def experiment_loop(methodID_string, experiment_IDs_collection, ModelSet, 
                    num_epochs = 500 , path_to_results = './' ):
    for experiment_number in experiment_IDs_collection:
        
        exp_ = exps.Experiment_Graph(gfds_name, 
                                     methodID_string, 
                                     experiment_number, graphs)
        pathModel = pathRes + exp_.experiment_name + '.pt'
        exp_.training_preparation(ModelSet)
        # training 
        exp_.training_run(num_epochs)
        exp_.validate()
        torch.save(exp_.beast,pathModel)
        exp_.xperiment_save(pathRes, )
        
def validation_loop(methodID_string, experiment_IDs_collection, ModelSet, 
 path_to_results = './' ):
    for experiment_number in experiment_IDs_collection:
        exp_ = exps.Experiment_Graph(gfds_name, methodID_string, experiment_number, graphs)
        pathModel = pathRes + exp_.experiment_name + '.pt'
        pathIMG = pathRes + exp_.experiment_name + '_val.png'
        exp_.training_preparation(ModelSet)
        exp_.model.load_state_dict(torch.load(pathModel))
        exp_.model.eval()
        exp_.validate()
        exp_.validate_plot()
        plt.savefig(pathIMG)
        plt.close()
# %% training loop pro sage
experiment_loop('SAGE_RF2', experiments_IDs_0, SAGE0, num_epochs = 50, path_to_results = pathRes)
experiment_loop('GCN_RF2', experiments_IDs_0, GCN0, num_epochs = 50 , path_to_results = pathRes)
# %%
validation_loop('SAGE_RF2', experiments_IDs_0, SAGE0, path_to_results = pathRes)
validation_loop('GCN_RF2', experiments_IDs_0, GCN0, path_to_results = pathRes)
# %% design evaluation of VAULT

for experiment_number in experiment_IDs_collection:
    exp_ = exps.Experiment_Graph(gfds_name, methodID_string, experiment_number, graphs)
    pathModel = pathRes + exp_.experiment_name + '.pt'
    pathIMG = pathRes + exp_.experiment_name + '_val.png'
    exp_.training_preparation(ModelSet)
    exp_.model.load_state_dict(torch.load(pathModel))
    break

D= np.load(exp_.vault_name_numpy_file)


# print d1.get('key1')
# print d2.item().get('key2')
    # exp_.model.eval()
    # exp_.validate()
    # exp_.validate_plot()
    # plt.savefig(pathIMG)


    
