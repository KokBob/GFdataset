'''
zavery:
melo by se postupovat podle pipeliny []
[1] https://docs.dgl.ai/en/0.6.x/guide/training-node.html
[2] https://docs.dgl.ai/en/0.6.x/guide/data.html
https://docs.dgl.ai/en/0.6.x/guide/message.html#guide-message-passing
https://plotly.com/python/network-graphs/
https://docs.dgl.ai/en/0.6.x/new-tutorial/6_load_data.html
https://paperswithcode.com/task/graph-regression/codeless
https://docs.dgl.ai/en/0.6.x/guide/training-edge.html#guide-training-edge-classification
https://proceedings.neurips.cc//paper/2020/file/99cad265a1768cc2dd013f0e740300ae-Paper.pdf1
https://docs.dgl.ai/en/latest/guide/graph-feature.html#guide-graph-feature
https://www.tutorialspoint.com/How-to-convert-Python-Dictionary-to-a-list
'''
#import dgl
from dgl.dataloading import GraphDataLoader
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gnsfem as gf
from gnsfem import preprocessing
from gnsfem import visualization
import dgl
import torch as th
import plotly as py
import plotly.graph_objs as go
import plotly.express as px
import glob as gl
from gnsfem_dataset import preprocessing_dataset
import torch
version = '001_2D'
# %% inp file open 
file_name = r'beam-01.inp'  # 2D mesh
file1       = open(file_name,"r")
d = file1.readlines() 
#%% get lines number
line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = '*Nset, nset=Set-Part, generate')
d_nodes         = preprocessing.get_nodes_2D(file_name, line_elements,line_nodes)
d_elements      = preprocessing.get_edges(file_name, line_elements,line_nodes, line_end)
# %% graph 2D
G = preprocessing.create_graph_DGL_2D(d_nodes, d_elements)
# %%  draw via netoworkx
# nxg = G.to_networkx()
# nx.draw(nxg, with_labels = True) 
# %% coordinates put to Graphs in 2D
# G.ndata['x'] = th.ones(G.num_nodes(), )
G.ndata['labels'] 	= th.tensor(d_nodes['#Node'].values)
Xn 					= d_nodes['x'].values
Yn 					= d_nodes['y'].values
G.ndata['x'] 		= th.tensor(d_nodes['x'].values)
G.ndata['y'] 		= th.tensor(d_nodes['y'].values)
# %% see graph .. not relabeling as i want to
nxg = G.to_networkx()
nx.draw(nxg, with_labels = True)
#%% select output data exported via ABQ
path2dir    = r'D:\Abaqus\00_beam2D\dataset_beam'
content     = gl.glob(path2dir+ "/*.rpt")
#%% Dataset dictionary initialization
D = {}
for content_ in content:
    file_name = content_
    file1       = open(file_name,"r")
    d           = file1.readlines() 
    line_init, line_minimum =   preprocessing_dataset.get_lines_number(d)
    header_                 =   preprocessing_dataset.get_header(file_name, line_init)
    res_data                =   preprocessing_dataset.get_data(file_name, line_init, line_minimum, header_[0:-1])   
    D_name = content_.split('\\')[-1].split('.')[0]
    D[D_name] = {}
    D[D_name]['data'] = res_data
# %% pristup k jednotlivymu datasetu
# print(D[list(D.keys())[-1]])
# D[list(D.keys())[0]]
# %% splitting on test train data ... 
# actually individual dataset will take this mask
# G.ndata['train_mask'] = torch.zeros(33, dtype=torch.bool).bernoulli(0.6)
# %%
graphs = []
labels = []
# label_dict = {}
justPlot = []
for graph_id in D.keys():   
    
    
    #g = G takhle prekopirovani graphu .. to nejak nefunguje
    # asi je potreba to vytvorit znova
    g = preprocessing.create_graph_DGL_2D(d_nodes, d_elements)
    # takze ano... je to potreba 
    g.ndata['labels'] 	= th.tensor(d_nodes['#Node'].values)
    # Xn 					= d_nodes['x'].values
    # Yn 					= d_nodes['y'].values
    g.ndata['x'] 		= th.tensor(d_nodes['x'].values,dtype = torch.float)
    g.ndata['y'] 		= th.tensor(d_nodes['y'].values,dtype = torch.float)
      
    nodes_data = D[graph_id]['data']
    # del g.ndata['S.mises']
    nodes_features_np = nodes_data['U.U2'].to_numpy()
    nodes_features_np_ = nodes_data['S.Mises'].to_numpy()
    # print(nodes_features_np[1:7])
    g.ndata['U.U2'] = th.tensor(nodes_features_np, dtype = torch.float)
    g.ndata['S.mises'] = th.tensor(nodes_features_np, dtype = torch.float)
    
    # print(g.ndata['S.mises'][1:7])
    # g.ndata['label'] = int(graph_id.split('_')[1])
    p = g
    # print(p)
    # break
    # graphs.append(g)
    graphs.append(p)
    labels.append(graph_id)
    # if graph_id == 'framebeam_100': break
# %% validace hodnot v jednotlivem grafu
# for _ in graphs:
#     s =  _.ndata['S.mises']
#     o =  _.ndata['x']
#     print(o)
#     print(s)
# %% shuffle dataset 
# https://www.w3schools.com/python/ref_random_shuffle.asp
import random 
random.seed(42)
# graphs_shuffled = random.shuffle(graphs)
random.shuffle(graphs)
# %% poznamky pro list
from dgl.nn import SAGEConv
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
train_loader = GraphDataLoader(graphs[:50],shuffle=True, )
test_loader = GraphDataLoader(graphs[50:],shuffle=False, ) 
# %%
# train_loader = GraphDataLoader(graphs_shuffled[:50],shuffle=True, batch_size=8)
# test_loader = GraphDataLoader(graphs_shuffled[50:],shuffle=False, batch_size=1) 
# %%
# train_loader = GraphDataLoader(train_dataset, batch_size=8)
# test_loader = GraphDataLoader(test_dataset, batch_size=1)    
for batch in train_loader: break    
in_channels = batch.ndata['S.mises'].shape[0]
# out_channels = batch.ndata['S.mises'].shape[1]

# %%
# net = GraphSAGE(5, 16, 2)
model = SAGE(in_channels, 100, 33)

# loss_fn = torch.nn.BCELoss()
loss_fn = F.mse_loss
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # obscurly running back 
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) # converging quite monotonically but too slowly 
# %%
num_epochs =100
my_device = "cuda" if torch.cuda.is_available() else "cpu"    
model = model.to(my_device)  
losses = []
losses_val = []
time_elapsed = []
epochs = []
inputs = 'U.U2'
targets = 'S.mises'
# t0 = time.time()    
for epoch in range(num_epochs):        
    total_loss = 0.0
    batch_count = 0        
    for batch in train_loader:            
            optimizer.zero_grad()
            batch = batch.to(my_device)
            pred = model(batch, batch.ndata[inputs].to(my_device))
            loss = loss_fn(pred, batch.ndata[targets].to(my_device))
            loss.backward()
            optimizer.step()            
            total_loss += loss.detach()
            batch_count += 1        
            mean_loss = total_loss / batch_count
    losses.append(mean_loss)
    epochs.append(epoch)
    # time_elapsed.append(time.time() - t0)        
    # if epoch % 100 == 0:
    if epoch % 5 == 1:
        print(f"loss at epoch {epoch} = {mean_loss}")    # get test accuracy score
    num_correct = 0.
    num_total = 0.
    model.eval()    
    
    for batch in test_loader:        
        batch = batch.to(my_device)
        pred_val = model(batch, batch.ndata[inputs])
        loss_val = loss_fn(pred_val, batch.ndata[targets].to(my_device))
        total_val = ((pred -batch.ndata[targets])/(pred -batch.ndata[targets]).sum()).sum()
        losses_val.append(loss_val.detach())
        # num_correct += (pred.round() ==  batch.ndata[targets].to(my_device)).sum()
        # num_total += pred.shape[0] * pred.shape[1]        
        # np.save("dgl.npy", 
            # {"epochs": epochs, \
            # "losses": losses, \
            # "time_elapsed": time_elapsed})    
        if epoch % 5 == 1:
            print(f"validation accuracy = {loss_val} , {total_val}")
        
# %%  
losses
import matplotlib.pyplot as plt
# plt.plot(losses)
plt.plot(losses_val)
plt.yscale("log")
