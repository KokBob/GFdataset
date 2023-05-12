# -*- coding: utf-8 -*-
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

from evaluation import ground_truth

GFDS = load_dataset.Beam3D()
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



        
def validation_loop(methodID_string, experiment_IDs_collection, ModelSet, 
 path_to_results = './' ):
    for experiment_number in experiment_IDs_collection:
        exp_ = exps.Experiment_Graph(gfds_name, methodID_string,
                                     experiment_number, graphs, pathRes )
        pathModel = pathRes + exp_.experiment_name + '.pt'
        pathIMG = pathRes + '/pics/' + exp_.experiment_name + '_val.png'
        exp_.training_preparation(ModelSet)
        exp_.model.load_state_dict(torch.load(pathModel))
        exp_.model.eval()
        exp_.validate()
        exp_.validate_plot()
        plt.savefig(pathIMG)
        plt.close()

# validation_loop('SAGE_RF2', experiments_IDs_0, SAGE0, path_to_results = pathRes)
# validation_loop('GCN_RF2', experiments_IDs_0, GCN0, path_to_results = pathRes)


GT_b3 = ground_truth.GroundTruth()
GT_b3.beam3D(vtu_file   = 'beam3D2.vtu')
# %%
for experiment_number in experiments_IDs_0:
    exp_ = exps.Experiment_Graph(gfds_name, 'SAGE_RF2',
                                 experiment_number, graphs, pathRes )
    pathModel = pathRes + exp_.experiment_name + '.pt'
    pathIMG = pathRes + '/pics/' + exp_.experiment_name + '_val.png'
    exp_.training_preparation(SAGE0)
    exp_.model.load_state_dict(torch.load(pathModel))
    exp_.model.eval()
    exp_.validate()
    break
# %%
exp_.validation_sample( magic_number = 0)
GT_b3.attach_result_fields(exp_.sample_x, exp_.sample_y, 
                           exp_.sample_y_hat, exp_.sample_err,
                           )
GT_b3.write_results_to_vtu()
    # exp_.validate_plot()
    # plt.savefig(pathIMG)
    # plt.close()
