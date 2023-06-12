# -*- coding: utf-8 -*-
import sys
import os 
import pandas as pd
sys.path.append("..") 
from load_dataset import load_dataset 
from modelling.modelsStore import GCN0
from modelling.modelsStore import SAGE0
from modelling.modelsStore import LR0
from modelling.modelsStore import SimpleNet
from modelling.preprocessing import *
import modelling.experimentationing as exps
import torch
import matplotlib.pyplot as plt

from evaluation import ground_truth
# %%

# GFDS = load_dataset.Beam2D()
# GFDS = load_dataset.Beam3D()
# GFDS = load_dataset.Fibonacci()
GFDS = load_dataset.Plane()

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
experiments_IDs_0 = [42,17,23,11,18,4,5,1,6,1212] # Not scaled

# %%
SET = [[ LR0, 'LR_RF2'],
[ SimpleNet, 'FN_RF2'],
[ GCN0, 'GCN_RF2'],
[ SAGE0, 'SAGE_RF2']]


# SET = [[ LR0, 'LR_RF3'],
# [ SimpleNet, 'FN_RF3'],
# [ GCN0, 'GCN_RF3'],
# [ SAGE0, 'SAGE_RF3']]


for _ in SET:
    
    MODEL, methodID_string = _
    
    # %%
    # MODEL, methodID_string = LR0, 'LR_RF2'
    # MODEL, methodID_string = SimpleNet, 'FN_RF2'
    # MODEL, methodID_string = GCN0, 'GCN_RF2'
    # MODEL, methodID_string = SAGE0, 'SAGE_RF2'
    # %%
    for experiment_number in experiments_IDs_0:
        if MODEL == (LR0 ):
        
            exp_ = exps.Experiment_ML(gfds_name, methodID_string, experiment_number, X0,y,  pathRes)      # ML
        elif MODEL == (SimpleNet):
            exp_ = exps.Experiment_ML(gfds_name, methodID_string, experiment_number, X0,y,  pathRes)      # ML
        else:
            exp_ = exps.Experiment_Graph(gfds_name, methodID_string, experiment_number, graphs, pathRes )   # Graphs 
        pathModel = pathRes + exp_.experiment_name + '.pt'
        pathIMG = pathRes + '/pics/' + exp_.experiment_name + '_val.png'
        exp_.training_preparation(MODEL)
        exp_.model.load_state_dict(torch.load(pathModel))
        exp_.model.eval()
        exp_.validate()
        break
    # %%
    plt.close()
    # exp_.validate_plot()    
    
    exp_.validate_skedacity_plot(title_of_experiment = ' '.join(exp_.experiment_name.split('_')[0:-2]))
    plt.savefig(f'{exp_.experiment_name}.png')
