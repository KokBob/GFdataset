# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import torch
# class Experiment(nn.Module):
#     def __init__(self, in_feats, hid_feats, out_feats):
#         super().__init__()
#         self.conv1 = dglnn.GraphConv(in_feats, hid_feats, norm='both', weight=True, bias=True)
#         # self.conv12 = dglnn.GraphConv(in_feats, hid_feats, norm='both', weight=True, bias=True)
#         self.conv2 = dglnn.SAGEConv(hid_feats, out_feats, aggregator_type='mean')

#     def forward(self, graph, inputs):
#         # inputs are features of nodes
#         h = self.conv1(graph, inputs)
#         h = F.relu(h)
#         # h = self.conv12(graph, inputs)
#         # h = F.relu(h)
#         h = self.conv2(graph, h)
#         return h
def experiment_evaluation(experiment_name,
                     pathRes,
                     model,
                     epochs,
                     time_elapsed,
                     losses_, 
                     losses_val_,
                     model_store = False):
    # epoch evaluation method
    # losses_fromEpoch
    # needs 
    L  = np.zeros([len(losses_)])
    LV = np.zeros([len(losses_)])
    for i in range(len(losses_)):
        L[i] = losses_[i].cpu().numpy()
        LV[i] = losses_val_[i].cpu().numpy()
    np.save(pathRes + experiment_name + ".npy", 
        {"epochs": epochs, \
        "losses": L, \
        "losses_val": LV, \
        "time_elapsed": time_elapsed})  
    if model_store: # for early bird 
        pathModel = pathRes + experiment_name + '.pt'
        torch.save(model.state_dict(),pathModel)

    plt.figure()
    plt.plot(L)
    plt.plot(LV) 
    plt.yscale("log")
    plt.savefig(pathRes + experiment_name + ".jpg")
    plt.close()