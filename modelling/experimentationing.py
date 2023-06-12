# -*- coding: utf-8 -*-
# update 090523
import sys
import os 
import pandas as pd
sys.path.append("..") 
import numpy as np
import matplotlib.pyplot as plt
import torch
import random 
from dgl.dataloading import GraphDataLoader
import time
import torch.nn.functional as F
from modelling.modelsStore import ds_splitting
import seaborn as sns
sns.set_style("whitegrid")
sns.set_palette("Spectral")
sns.color_palette("Blues", as_cmap=True)

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


class Experiment_ML(object):
    def __init__(self,        
                 gfds_name, 
                 methodID, 
                 experiment_number,
                 GFDS_X0, GFDS_y,
                 # graphs,
                 pathRes = './'
                 ):
        self.experiment_name        = f'{gfds_name}_{methodID}_{experiment_number}'      
        self.pathModel              = pathRes + self.experiment_name + '.pt'
        self.vault_name_numpy_file  = pathRes + self.experiment_name + '.npy'
        
        random.seed(experiment_number)

        
        self.train_loader, self.test_loader = ds_splitting(GFDS_X0,GFDS_y)
        self.GFDS_X0,self.GFDS_y = GFDS_X0,GFDS_y
        # self.train_loader = GraphDataLoader(graphs[:split_number],shuffle=True, )
        # self.test_loader  = GraphDataLoader(graphs[split_number:],shuffle=False, ) 
    def training_preparation(self, MODEL):
        self.VAULT = {}
        # self.loss_fn         = F.mse_loss
        # if not LOSS_FUNCTION:
        self.loss_fn         = F.mse_loss
        # else:  self.loss_fn         = LOSS_FUNCTION
        self.loss_fn_rmax    = root_max_square_error
        self.my_device   = "cuda" if torch.cuda.is_available() else "cpu"  
        
        for batch in self.train_loader: break    
        in_channels, out_channels     = self.GFDS_X0.shape[1],self.GFDS_y.shape[1]

        
        self.model      = MODEL(in_channels, out_channels )
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model      = self.model.to(self.my_device)  
        
        self.losses      = []
        self.losses_rmax = []
        
        self.losses_val  = []
        
        self.time_elaps  = []
        self.epochs      = []
        self.inputs      = 'x'
        self.targets     = 'y'
        self.t0          = time.time()    
    def validate_plot(self, ):
        fig, axs = plt.subplots(2, 3)
        # for batch in self.test_loader:   
        for xbv,ybv in self.test_loader:  
            # batch = batch.to(self.my_device)
            pred_val        = self.model(xbv.to(self.my_device))
            
            x               = xbv.cpu().numpy()
            y               = ybv.cpu().numpy()
            y_hat           = pred_val.cpu().detach().numpy()
            
            err     = y - y_hat
            err_avg = err/len(err)
            self.err_val = err
            
            
            axs[0, 0].scatter(x,        y,         c=err, alpha=0.5)
            axs[0, 0].set_title(r'$y(X)$')
            
            axs[0, 1].scatter(x,        y_hat,     c=err, alpha=0.5)
            axs[0, 1].set_title(r'$\hat{y}(X)$')
            
            axs[1, 0].scatter(y_hat,    err,       c=err, alpha=0.5)
            axs[1, 0].set_title(r'$\hat{y}(err)$')
            
            axs[1, 1].scatter(y,        y_hat,     c=err, alpha=0.5)
            axs[1, 1].set_title(r'$\hat{y}(y)$')
            # axs[1, 1].set_xaxis(r'$\hat{y}(y)$')
        
            axs[1, 2].scatter(x,        err,     c=err, alpha=0.5)
            axs[1, 2].set_title(r'$err(x)$')
            
            axs[0, 2].scatter(y_hat,    err_avg,       c=err, alpha=0.5)
            axs[0, 2].set_title(r'$\hat{y}(err_avg)$')
        # plt.tight_layout()

    def validate_skedacity_plot(self, title_of_experiment):
        # https://stackoverflow.com/questions/15908371/matplotlib-colorbars-and-its-text-labels
        # https://matplotlib.org/2.2.5/gallery/api/agg_oo_sgskip.html
        # plt.figure(figsize=[10,3])
        alpha_set = 0.9
        fig, axs = plt.subplots(1, 2, figsize=(10, 5),)
        # fig, axs = plt.subplots(1, 3, figsize=(10, 5),)
                                # constrained_layout=True)
        # fig, axs = plt.subplots(1, 4)
        fig1 = axs[0]
        fig2 = axs[1]
        # fig3 = axs[2]
        # fig4 = axs[3]
        fig4 = plt.axes([0.1, 0.05, 0.97, 0.850])
        
        # fig, axs = plt.subplots(2, 3)
        # fig1 = axs[0,0]
        # fig2 = axs[0,1]
        # fig3 = axs[0,2]
        # fig.figsize([10,2])
        
        # for batch in self.test_loader:   
        for xbv,ybv in self.test_loader:  
            # batch = batch.to(self.my_device)
            pred_val        = self.model(xbv.to(self.my_device))
            
            x               = xbv.cpu().numpy()
            y               = ybv.cpu().numpy()
            y_hat           = pred_val.cpu().detach().numpy()
            
            err     = y - y_hat
            err_avg = err/len(err)
            self.err_val = err
            

            
            # fig4 = axs[3]
            
            fig1.scatter(y,        y_hat,     c=err, alpha=alpha_set)
            fig1.set_title(r'$\hat{y}(y)$')
            fig1.set_xlabel(r'$y(\sigma) \:  [MPa]$')
            fig1.set_ylabel(r'$\hat{y}(\sigma) \:  [MPa]$')

            
            # sc3 = fig3.scatter(y_hat,    err_avg,       c=err, alpha=alpha_set)
            sc2 = fig2.scatter(y,err,c=err, alpha=alpha_set)
            fig2.set_title(r'$e(y)$')
            fig2.set_xlabel(r'$y(\sigma) \:  [MPa]$')
            fig2.set_ylabel(r'$e(\sigma) \:  [MPa]$')

            
            # sc3 = fig3.scatter(y_hat,    err_avg,       c=err, alpha=alpha_set)
            # fig3.set_title(r'$e(\hat{y})$')
            # fig3.set_xlabel(r'$\hat{y}(\sigma) \:  [MPa]$')
            # fig3.set_ylabel(r'$e(\sigma) \:  [MPa]$')
            
            
            # fig3.colorbar() # nf
        # fig4 = plt.axes([0.1, 0, 0.8, 0.01]
        # clb=plt.colorbar()
        # cax = plt.axes([0.1, 0, 0.8, 0.01]) #Left,bottom, length, width
        # clb=plt.colorbar(cax=cax,orientation="horizontal")
        # clb.ax.tick_params(labelsize=8) 
        # clb.ax.set_title('Your Label',fontsize=8)
        
        # clb= fig.colorbar(sc3, ax=fig4, orientation="horizontal", location="bottom")
        clb= fig.colorbar(sc2, ax=fig4, orientation="vertical", )
        # clb= fig.colorbar(sc3, ax=fig4, orientation="vertical", )
        # clb= fig.colorbar(sc3, ax=fig3, orientation="vertical", )
        clb.ax.tick_params(labelsize=10) 
        clb.ax.set_title(r'$e \: [MPa]$',fontsize=10,  ) # rotation=270,
        
        fig4.grid(False)
        fig4.axis('off')
        
        # plt.tight_layout()   
        plt.title(f'{title_of_experiment}\n')


    def validate(self, ):
        # for batch in self.test_loader: 
        for xbv,ybv in self.test_loader:  
            total_loss      = 0.0
            batch_count     = 0      
            total_loss_val  = 0.0
            batch_count_val = 0 
            
            
            pred_val        = self.model(xbv.to(self.my_device))
            self.loss_val   = self.loss_fn(pred_val, ybv.to(self.my_device))
            
            
            x               = xbv.cpu().numpy()
            y               = ybv.cpu().numpy()
            y_hat           = pred_val.cpu().detach().numpy()
            self.err_val    = y - y_hat
            
            
            total_loss  += self.loss_val.detach()
            batch_count += 1       
            
            mean_loss   = total_loss / batch_count
        print(f"Validation loss = {mean_loss}")
        
    def validation_sample(self, magic_number):
        i = 0
        # for batch in self.test_loader: 
        for xbv,ybv in self.test_loader:
            if i == magic_number:
                break
            else: i+=1
        total_loss      = 0.0
        batch_count     = 0      
        total_loss_val  = 0.0
        batch_count_val = 0 
            

        pred_val        = self.model(xbv.to(self.my_device))
        self.loss_val   = self.loss_fn(pred_val, ybv.to(self.my_device))
            
            
        self.sample_x               = xbv.cpu().cpu().numpy()
        self.sample_y               = ybv.cpu().cpu().numpy()
        self.sample_y_hat           = pred_val.cpu().detach().numpy()
        self.sample_err              = self.sample_y - self.sample_y_hat
        
        
        # total_loss  += self.loss_val.detach()
        # batch_count += 1       
        
        # mean_loss   = total_loss / batch_count
        # print(f"Validation loss = {mean_loss}")
        # print(f"validate loss at epoch {self.epoch} = {mean_loss}")
            # print(self.err_val.sum())
            # print(err.sum())
    def training_run(self, num_epochs):
        self.best       = 10000
        self.beast      = self.model.state_dict()
        for epoch in range(num_epochs):        
            self.epoch = epoch
            total_loss      = 0.0
            batch_count     = 0      
            total_loss_val  = 0.0
            batch_count_val = 0 
            
            # for batch in self.train_loader:  graphy 
            for xb,yb in self.train_loader: 
                
                self.optimizer.zero_grad()
                
                pred = self.model(xb.to(self.my_device))
                
                self.loss       = self.loss_fn(pred, yb.to(self.my_device))
                self.loss_rmax  = self.loss_fn_rmax(pred, yb.to(self.my_device))
                
                
                self.loss.backward()
                self.optimizer.step()            
                total_loss += self.loss.detach()
                batch_count += 1        
                mean_loss = total_loss / batch_count
                self.losses.append(mean_loss)
                self.epochs.append(epoch)
                self.time_elaps.append(time.time() - self.t0)        
            if epoch % 5 == 1:
                # pass
                print(f"loss at epoch {epoch} = {mean_loss}")    # get test accuracy score
                
            num_correct = 0.
            num_total = 0.
            self.model.eval()    
            self.validate()
            threasure_value = np.abs( self.err_val.sum() )
            node_error_avg = threasure_value / len( self.err_val )
            if threasure_value  <= self.best:
                self.best = self.err_val.sum()
                print(f"Beast Validation error sum {self.err_val.sum()}")
                print(f"Beast Validation error threasure {threasure_value}")
                print(f"Beast Validation error on node {node_error_avg}")
                self.beast = self.model.state_dict()
                threasure_value = self.best
            
            self.beast = self.beast 
            
            self.VAULT[epoch] = {}
            self.VAULT[epoch]['SumErr']         = self.err_val.sum()
            self.VAULT[epoch]['BeastVar']       = self.best
            # self.VAULT[epoch]['threasure']      = threasure_value
            self.VAULT[epoch]['NodeErr']        = total_loss
            self.VAULT[epoch]['TotalLossVal']   = total_loss_val
            self.VAULT[epoch]['loss_Rmax']      = self.loss_rmax.detach()
            
    def xperiment_save(self,path_to_results_folder_string):
        self.vault_name_numpy_file = path_to_results_folder_string +  self.experiment_name + '.npy'
        np.save(self.vault_name_numpy_file, self.VAULT)
        # https://stackoverflow.com/questions/40219946/python-save-dictionaries-through-numpy-save
        pass
    def xperiment_load(self,):
        self.vault_name_numpy_file = path_to_results_folder_string +  self.experiment_name + '.npy'
        # np.save(self.vault_name_numpy_file, self.VAULT)
        
        D= np.load(self.vault_name_numpy_file)
        return D
        # https://stackoverflow.com/questions/40219946/python-save-dictionaries-through-numpy-save
class Experiment_Graph(object):
    def __init__(self,        
                 gfds_name, 
                 methodID, 
                 experiment_number,
                 graphs,
                 pathRes = './'
                 ):
        self.experiment_name        = f'{gfds_name}_{methodID}_{experiment_number}'      
        self.pathModel              = pathRes + self.experiment_name + '.pt'
        self.vault_name_numpy_file  = pathRes + self.experiment_name + '.npy'
        
        random.seed(experiment_number)
        random.shuffle(graphs)

        split_number = int(len(graphs)*.7)
        self.train_loader = GraphDataLoader(graphs[:split_number],shuffle=True, )
        self.test_loader  = GraphDataLoader(graphs[split_number:],shuffle=False, ) 
    def training_preparation(self, MODEL):
        self.VAULT = {}
        # self.loss_fn         = F.mse_loss
        # if not LOSS_FUNCTION:
        self.loss_fn         = F.mse_loss
        # else:  self.loss_fn         = LOSS_FUNCTION
        self.loss_fn_rmax    = root_max_square_error
        self.my_device   = "cuda" if torch.cuda.is_available() else "cpu"    
        for batch in self.train_loader: break    
        in_channels     = batch.ndata['y'].shape[0]
        self.model      = MODEL(in_channels, in_channels, in_channels)
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model      = self.model.to(self.my_device)  
        
        self.losses      = []
        self.losses_rmax = []
        
        self.losses_val  = []
        
        self.time_elaps  = []
        self.epochs      = []
        self.inputs      = 'x'
        self.targets     = 'y'
        self.t0          = time.time()    
    def validate_plot(self, ):
        fig, axs = plt.subplots(2, 3)
        for batch in self.test_loader:   
            batch = batch.to(self.my_device)
            pred_val = self.model(batch, batch.ndata[self.inputs])
            
            x       = batch.ndata[self.inputs].cpu().numpy()
            y       = batch.ndata[self.targets].cpu().numpy()
            y_hat   = pred_val.cpu().detach().numpy()
            err     = y - y_hat
            err_avg = err/len(err)
            self.err_val = err
            
            
            axs[0, 0].scatter(x,        y,         c=err, alpha=0.5)
            axs[0, 0].set_title(r'$y(X)$')
            
            axs[0, 1].scatter(x,        y_hat,     c=err, alpha=0.5)
            axs[0, 1].set_title(r'$\hat{y}(X)$')
            
            axs[1, 0].scatter(y_hat,    err,       c=err, alpha=0.5)
            axs[1, 0].set_title(r'$\hat{y}(err)$')
            
            axs[1, 1].scatter(y,        y_hat,     c=err, alpha=0.5)
            axs[1, 1].set_title(r'$\hat{y}(y)$')
            # axs[1, 1].set_xaxis(r'$\hat{y}(y)$')
        
            axs[1, 2].scatter(x,        err,     c=err, alpha=0.5)
            axs[1, 2].set_title(r'$err(x)$')
            
            axs[0, 2].scatter(y_hat,    err_avg,       c=err, alpha=0.5)
            axs[0, 2].set_title(r'$\hat{y}(err_avg)$')
        plt.tight_layout()
    def validate_skedacity_plot(self, title_of_experiment):
        # https://stackoverflow.com/questions/15908371/matplotlib-colorbars-and-its-text-labels
        # https://matplotlib.org/2.2.5/gallery/api/agg_oo_sgskip.html
        # plt.figure(figsize=[10,3])
        alpha_set = 0.9
        # fig, axs = plt.subplots(1, 3, figsize=(10, 5),)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5),)
                                # constrained_layout=True)
        # fig, axs = plt.subplots(1, 4)
        fig1 = axs[0]
        fig2 = axs[1]
        # fig3 = axs[2]
        # fig4 = axs[3]
        fig4 = plt.axes([0.1, 0.05, 0.97, 0.850])
        
        # fig, axs = plt.subplots(2, 3)
        # fig1 = axs[0,0]
        # fig2 = axs[0,1]
        # fig3 = axs[0,2]
        # fig.figsize([10,2])
        
        # for batch in self.test_loader:   
        for batch in self.test_loader:   
            batch = batch.to(self.my_device)
            pred_val = self.model(batch, batch.ndata[self.inputs])
            
            x       = batch.ndata[self.inputs].cpu().numpy()
            y       = batch.ndata[self.targets].cpu().numpy()
            y_hat   = pred_val.cpu().detach().numpy()
            err     = y - y_hat
            err_avg = err/len(err)
            self.err_val = err
            

            
            # fig4 = axs[3]
            
            fig1.scatter(y,        y_hat,     c=err, alpha=alpha_set)
            fig1.set_title(r'$\hat{y}(y)$')
            fig1.set_xlabel(r'$y(\sigma) \:  [MPa]$')
            fig1.set_ylabel(r'$\hat{y}(\sigma) \:  [MPa]$')
            




            sc2 = fig2.scatter(y,err,c=err, alpha=alpha_set)
            fig2.set_title(r'$e(y)$')
            fig2.set_xlabel(r'$y(\sigma) \:  [MPa]$')
            fig2.set_ylabel(r'$e(\sigma) \:  [MPa]$')
            fig2.set_yticks(fig2.get_yticks())
            fig2.set_yticklabels(fig2.get_yticklabels(), rotation=90, )
                                 # ha='right') 
            
            # sc3 = fig3.scatter(y_hat,    err_avg,       c=err, alpha=alpha_set)
            # fig3.set_title(r'$e(y)$')
            # fig3.set_xlabel(r'$\hat{y}(\sigma) \:  [MPa]$')
            # fig3.set_ylabel(r'$e(\sigma) \:  [MPa]$')
            # fig3.set_yticks(fig3.get_yticks())
            # fig3.set_yticklabels(fig3.get_yticklabels(), rotation=45, )
                                 # ha='right')
            
            # fig3.colorbar() # nf
        # fig4 = plt.axes([0.1, 0, 0.8, 0.01]
        # clb=plt.colorbar()
        # cax = plt.axes([0.1, 0, 0.8, 0.01]) #Left,bottom, length, width
        # clb=plt.colorbar(cax=cax,orientation="horizontal")
        # clb.ax.tick_params(labelsize=8) 
        # clb.ax.set_title('Your Label',fontsize=8)
        
        # clb= fig.colorbar(sc3, ax=fig4, orientation="horizontal", location="bottom")
        clb= fig.colorbar(sc2, ax=fig4, orientation="vertical", )
        # clb= fig.colorbar(sc3, ax=fig4, orientation="vertical", )
        # clb= fig.colorbar(sc3, ax=fig3, orientation="vertical", )
        clb.ax.tick_params(labelsize=10) 
        clb.ax.set_title(r'$e \: [MPa]$',fontsize=10,  ) # rotation=270,
        
        fig4.grid(False)
        fig4.axis('off')
        
        # plt.tight_layout()   
        plt.title(f'{title_of_experiment}\n')
        # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.2)

        
    def validate(self, ):
        for batch in self.test_loader:   
            total_loss      = 0.0
            batch_count     = 0      
            total_loss_val  = 0.0
            batch_count_val = 0 
            
            batch       = batch.to(self.my_device)
            pred_val    = self.model(batch, batch.ndata[self.inputs])
            self.loss_val   = self.loss_fn(pred_val, batch.ndata[self.targets].to(self.my_device))
            
            
            x               = batch.ndata[self.inputs].cpu().numpy()
            y               = batch.ndata[self.targets].cpu().numpy()
            y_hat           = pred_val.cpu().detach().numpy()
            self.err_val    = y - y_hat
            
            
            total_loss  += self.loss_val.detach()
            batch_count += 1       
            
            mean_loss   = total_loss / batch_count
        print(f"Validation loss = {mean_loss}")
        
    def validation_sample(self, magic_number):
        i = 0
        for batch in self.test_loader: 
            if i == magic_number:
                break
            else: i+=1
        total_loss      = 0.0
        batch_count     = 0      
        total_loss_val  = 0.0
        batch_count_val = 0 
            
        batch       = batch.to(self.my_device)
        pred_val    = self.model(batch, batch.ndata[self.inputs])
        self.loss_val   = self.loss_fn(pred_val, batch.ndata[self.targets].to(self.my_device))
            
            
        self.sample_x               = batch.ndata[self.inputs].cpu().numpy()
        self.sample_y               = batch.ndata[self.targets].cpu().numpy()
        self.sample_y_hat           = pred_val.cpu().detach().numpy()
        self.sample_err              = self.sample_y - self.sample_y_hat
        
        
        # total_loss  += self.loss_val.detach()
        # batch_count += 1       
        
        # mean_loss   = total_loss / batch_count
        # print(f"Validation loss = {mean_loss}")
        # print(f"validate loss at epoch {self.epoch} = {mean_loss}")
            # print(self.err_val.sum())
            # print(err.sum())
    def training_run(self, num_epochs):
        self.best       = 10000
        self.beast      = self.model.state_dict()
        for epoch in range(num_epochs):        
            self.epoch = epoch
            total_loss      = 0.0
            batch_count     = 0      
            total_loss_val  = 0.0
            batch_count_val = 0 
            
            for batch in self.train_loader: 
                
                self.optimizer.zero_grad()
                batch = batch.to(self.my_device)
                pred = self.model(batch, batch.ndata[self.inputs].to(self.my_device))
                
                self.loss       = self.loss_fn(pred, batch.ndata[self.targets].to(self.my_device))
                self.loss_rmax  = self.loss_fn_rmax(pred, batch.ndata[self.targets].to(self.my_device))
                
                
                self.loss.backward()
                self.optimizer.step()            
                total_loss += self.loss.detach()
                batch_count += 1        
                mean_loss = total_loss / batch_count
                self.losses.append(mean_loss)
                self.epochs.append(epoch)
                self.time_elaps.append(time.time() - self.t0)        
            if epoch % 5 == 1:
                # pass
                print(f"loss at epoch {epoch} = {mean_loss}")    # get test accuracy score
                
            num_correct = 0.
            num_total = 0.
            self.model.eval()    
            self.validate()
            threasure_value = np.abs( self.err_val.sum() )
            node_error_avg = threasure_value / len( self.err_val )
            if threasure_value  <= self.best:
                self.best = self.err_val.sum()
                print(f"Beast Validation error sum {self.err_val.sum()}")
                print(f"Beast Validation error threasure {threasure_value}")
                print(f"Beast Validation error on node {node_error_avg}")
                self.beast = self.model.state_dict()
                threasure_value = self.best
            
            self.beast = self.beast 
            
            self.VAULT[epoch] = {}
            self.VAULT[epoch]['SumErr']         = self.err_val.sum()
            self.VAULT[epoch]['BeastVar']       = self.best
            # self.VAULT[epoch]['threasure']      = threasure_value
            self.VAULT[epoch]['NodeErr']        = total_loss
            self.VAULT[epoch]['TotalLossVal']   = total_loss_val
            self.VAULT[epoch]['loss_Rmax']      = self.loss_rmax.detach()
            
    def xperiment_save(self,path_to_results_folder_string):
        self.vault_name_numpy_file = path_to_results_folder_string +  self.experiment_name + '.npy'
        np.save(self.vault_name_numpy_file, self.VAULT)
        # https://stackoverflow.com/questions/40219946/python-save-dictionaries-through-numpy-save
        pass
    def xperiment_load(self,):
        self.vault_name_numpy_file = path_to_results_folder_string +  self.experiment_name + '.npy'
        # np.save(self.vault_name_numpy_file, self.VAULT)
        
        D= np.load(self.vault_name_numpy_file)
        return D
        # https://stackoverflow.com/questions/40219946/python-save-dictionaries-through-numpy-save
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
