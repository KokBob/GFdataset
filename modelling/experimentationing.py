# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import torch
import random 
from dgl.dataloading import GraphDataLoader
import time
import torch.nn.functional as F

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
def root_max_square_error(pred, target):
    error = torch.abs(pred - target)
    max_error = torch.max(error)
    return torch.sqrt(max_error**2)

class Experiment_Graph(object):
    def __init__(self,        
                 gfds_name, 
                 methodID, 
                 experiment_number,
                 graphs,
                 ):
        self.experiment_name = f'{gfds_name}_{methodID}_{experiment_number}'
        random.seed(experiment_number)
        random.shuffle(graphs)

        split_number = int(len(graphs)*.7)
        self.train_loader = GraphDataLoader(graphs[:split_number],shuffle=True, )
        self.test_loader  = GraphDataLoader(graphs[split_number:],shuffle=False, ) 
    def training_preparation(self, MODEL):
        # self.loss_fn         = F.mse_loss
        # if not LOSS_FUNCTION:
        #     self.loss_fn         = F.mse_loss
        # else:  self.loss_fn         = LOSS_FUNCTION
        self.loss_fn         = root_max_square_error
        self.my_device   = "cuda" if torch.cuda.is_available() else "cpu"    
        for batch in self.train_loader: break    
        in_channels     = batch.ndata['y'].shape[0]
        self.model      = MODEL(in_channels, in_channels, in_channels)
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model      = self.model.to(self.my_device)  
        
        self.losses      = []
        self.losses_val  = []
        self.time_elaps  = []
        self.epochs      = []
        self.inputs      = 'x'
        self.targets     = 'y'
        self.t0          = time.time()    
    def validate_plot(self, ):
        for batch in self.test_loader:   
            batch = batch.to(self.my_device)
            pred_val = self.model(batch, batch.ndata[self.inputs])
            
            x       = batch.ndata[self.inputs].cpu().numpy()
            y       = batch.ndata[self.targets].cpu().numpy()
            y_hat   = pred_val.cpu().detach().numpy()
            self.err_val     = y - y_hat
    def validate(self, ):
        for batch in self.test_loader:   
            batch       = batch.to(self.my_device)
            pred_val    = self.model(batch, batch.ndata[self.inputs])
            
            x               = batch.ndata[self.inputs].cpu().numpy()
            y               = batch.ndata[self.targets].cpu().numpy()
            y_hat           = pred_val.cpu().detach().numpy()
            self.err_val    = y - y_hat
            print(self.err_val.sum())
            # print(err.sum())
    def training_run(self, num_epochs):
        self.best       = 10000
        for epoch in range(num_epochs):        
            total_loss      = 0.0
            batch_count     = 0      
            total_loss_val  = 0.0
            batch_count_val = 0 
            
            for batch in self.train_loader:            
                self.optimizer.zero_grad()
                batch = batch.to(self.my_device)
                pred = self.model(batch, batch.ndata[self.inputs].to(self.my_device))
                
                self.loss = self.loss_fn(pred, batch.ndata[self.targets].to(self.my_device))
                self.loss = self.loss_fn(pred, batch.ndata[self.targets].to(self.my_device))
                
                self.loss.backward()
                self.optimizer.step()            
                total_loss += self.loss.detach()
                batch_count += 1        
                mean_loss = total_loss / batch_count
                self.losses.append(mean_loss)
                self.epochs.append(epoch)
                self.time_elaps.append(time.time() - self.t0)        
            if epoch % 5 == 1:
                print(f"loss at epoch {epoch} = {mean_loss}")    # get test accuracy score
                
            num_correct = 0.
            num_total = 0.
            self.model.eval()    
            self.validate()
            if self.err_val.sum() <= self.best:
                self.best = self.err_val.sum()
                print(f"Validation error {self.err_val.sum()}")
                self.beast = self.model.state_dict()
        def xperiment_save():
            pass
class Evaluation(object):
    def __init__(self,        
                 gfds_name, 
                 methodID, 
                 experiment_number,
                 graphs,
                 ): 
        pass
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
