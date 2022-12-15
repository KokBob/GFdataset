# import getData
import dgl
import torch 
import networkx as nx
import torch.nn as nn
# G, x, y = getData.getCRdata()
# %%
x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float)
y = torch.tensor([0, 1, 0, 1], dtype=torch.float)
edge_index = torch.tensor([[0, 2, 1, 0, 3],
                           [3, 1, 0, 1, 2]], dtype=torch.long)
# %% create graph based on edges [r0]
src_ids = edge_index[0]
dst_ids = edge_index[1]
g = dgl.graph((src_ids, dst_ids))
u, v = g.edges()
# %% 
gnx = dgl.to_networkx(g)
nx.draw_networkx(gnx)
# train_loader = GraphDataLoader(train_dataset, batch_size=8)
# %%
# DATA
# data = dgl.data(x=x, y=y, edge_index=edge_index)
# ids = []
# S = [] 
# F = [] # pouze zatezujici silu
# for nid, attr in G.nodes(data=True):
#     print("nid", nid)
#     print("attr", attr)
#     S.append(attr['S']) 
# %%
# To train a GNN for graph classification on a set of graphs in dataset (assume the backend is PyTorch):
# dataloader = dgl.dataloading.GraphDataLoader(
#     dataset, batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
# for batched_graph, labels in dataloader:
#     train_on(batched_graph, labels)
    # %%
# sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
# %%
# dataloader = dgl.dataloading.NodeDataLoader(
#     g, train_nid, sampler,
#     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
# %%
# for input_nodes, output_nodes, blocks in dataloader:
    # train_on(input_nodes, output_nodes, blocks)
# https://docs.dgl.ai/en/0.6.x/api/python/dgl.dataloading.html
# nx.set_node_attributes(G, S, "S")
# %% references
# -*- coding: utf-8 -*-
# [r0] https://docs.dgl.ai/en/0.9.x/generated/dgl.graph.html
# https://github.com/KokBob/PhysGNN/blob/f527de0b2b9405de8bf9934e459bfa68ddb2ff38/models.py#L1
# https://www.youtube.com/watch?v=-UjytpbqX4A
# https://stackoverflow.com/questions/68202388/graph-neural-network-regression
# https://stackoverflow.com/questions/72849912/regression-case-for-graph-neural-network
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html
# https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-1-d98dc93e7742
# https://pytorch.org/docs/stable/tensorboard.html
# https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8
# https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.set_node_attributes.html
# https://medium.com/@khang.pham.exxact/pytorch-geometric-vs-deep-graph-library-626ff1e802
