# import getData
import dgl
import torch 
import numpy as np
import networkx as nx
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader # co je dataloade2?
import torch.nn.functional as F
from dgl.nn import SAGEConv
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, num_outputs):
    # def __init__(self, in_feats, h_feats, num_outputs):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, 150, 'mean')
        self.conv2 = SAGEConv(150, num_outputs, 'mean')
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# G, x, y = getData.getCRdata()
# %%
xnp = np.array([[2,1], [5,6], [3,7], [12,0]])
x = torch.tensor(xnp, dtype=torch.float)
# x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float)
y = torch.tensor([0, 1, 0, 1], dtype=torch.float)
edge_index = torch.tensor([[0, 2, 1, 0, 3],
                           [3, 1, 0, 1, 2]], dtype=torch.long)
# %% create graph based on edges [r0]
src_ids = edge_index[0]
dst_ids = edge_index[1]
g = dgl.graph((src_ids, dst_ids))
u, v = g.edges()
# %% draw in nx 
gnx = dgl.to_networkx(g)
nx.draw_networkx(gnx)
# %%
print(g.nodes())
print(g.edges())
# %%
# g.add_nodes(len(x)) # uz tam nody jsou 
# g.ndata['node'] = torch.LongTensor(xnp[:,0],dtype=torch.float)
# g.ndata['node2'] = torch.LongTensor(xnp, dtype=torch.float)
# g.ndata['node'] = torch.LongTensor(xnp[:,0],dtype=torch.float)
# g.ndata['node2'] = torch.LongTensor(xnp, dtype=torch.float)
# %%
in_channels = 2
out_channels = 2
loss_f = F.mse_loss
num_epochs = 10
lr = 1e-3
my_seed = 42
model = GraphSAGE(in_channels, out_channels)
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in range(num_epochs):
    pred = model(g, g.ndata['node'])
    loss_f.backward()
    optimizer.step()
    

# %%
model = GraphSAGE(in_channels, out_channels)
in_channels = 2
out_channels = 1
pred = model(g, x)    
 
# %%
# https://discuss.dgl.ai/t/how-to-build-custom-graph-classification-datasets-in-dgl/1508/5
# https://www.kaggle.com/code/rpeer333/pytorch-dgl-schnet-a-graph-convolutional-net
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html
