# -*- coding: utf-8 -*-
import sys
import os 
import pandas as pd
sys.path.append("..") 
from load_dataset import load_dataset 
# from modelling.modelsStore import GCN0
# from modelling.modelsStore import SAGE0
from modelling.modelsStore import LR0
from modelling.modelsStore import SimpleNet
from modelling.preprocessing import *
import modelling.experimentationing as exps
import torch
import matplotlib.pyplot as plt

from evaluation import ground_truth


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

# methodID_string = 'LR_RF2'
# MODEL = LR0
# methodID_string = 'FN_RF2'
# MODEL = SimpleNet
GT = ground_truth.GroundTruth()
GT.plane(vtu_file   = GFDS.gfds_name + '_' + methodID_string + '_ml_01.vtu')
# %%

for experiment_number in experiments_IDs_0:
    exp_ = exps.Experiment_ML(gfds_name, methodID_string, experiment_number, X0,y,  pathRes)
    pathModel = pathRes + exp_.experiment_name + '.pt'
    pathIMG = pathRes + '/pics/' + exp_.experiment_name + '_val.png'
    # exp_.training_preparation(LR0)
    exp_.training_preparation(MODEL)
    # exp_.training_preparation(GCN0)
    exp_.model.load_state_dict(torch.load(pathModel))
    exp_.model.eval()
    exp_.validate()
    break
# %%
exp_.validation_sample( magic_number = 5)
GT.attach_result_fields(exp_.sample_x, exp_.sample_y, 
                           exp_.sample_y_hat, exp_.sample_err,
                           )
GT.write_results_to_vtu()
GT.Wizal()
app = GT.ViewHTML()
app.title = methodID_string
# app = GT.ViewHTML_error_only()
if __name__ == "__main__":    app.run_server(debug=True, port=8050)

