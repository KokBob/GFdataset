# -*- coding: utf-8 -*-

# ----------------------------- Configuration 1 --------------------------------
# -----------------------------                  --------------------------------
import dgl
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from dgl.nn import SAGEConv
import torch
import torch.nn.functional as F
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, num_classes, 'mean')
        self.linear1 = torch.nn.Linear(num_classes,num_classes) # regression
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = self.linear1(h) # regression
        return h

class config1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(config1, self).__init__()

        self.conv1 = GraphConv(input_dim, hidden_dim, aggr='max')
        self.conv2 = GraphConv(hidden_dim, hidden_dim, aggr='max')
        self.conv3 = GraphConv(hidden_dim, hidden_dim, aggr='max')
        self.conv4 = SAGEConv(hidden_dim, hidden_dim, aggr='max')
        self.conv5 = SAGEConv(hidden_dim, hidden_dim, aggr='max')
        self.conv6 = SAGEConv(hidden_dim, hidden_dim, aggr='max')

        # self.jk1 = JumpingKnowledge("lstm", hidden_dim, 3)
        # self.jk2 = JumpingKnowledge("lstm", hidden_dim, 3)

        self.lin1 = torch.nn.Linear(hidden_dim, 63)
        self.lin2 = torch.nn.Linear(63, 3)

        self.active1 = nn.PReLU(hidden_dim)
        self.active2 = nn.PReLU(hidden_dim)
        self.active3 = nn.PReLU(hidden_dim)
        self.active4 = nn.PReLU(hidden_dim)
        self.active5 = nn.PReLU(hidden_dim)
        self.active6 = nn.PReLU(hidden_dim)
        self.active7 = nn.PReLU(63)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_weight = 1 / edge_weight
        edge_weight = edge_weight.float()

        x = self.conv1(x, edge_index, edge_weight)
        x = self.active1(x)
        xs = [x]

        x = self.conv2(x, edge_index, edge_weight)
        x = self.active2(x)
        xs += [x]

        x = self.conv3(x, edge_index, edge_weight)
        x = self.active3(x)
        xs += [x]

        # ~~~~~~~~~~~~Jumping knowledge applied ~~~~~~~~~~~~~~~
        x = self.jk1(xs)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        x = self.conv4(x, edge_index)
        x = self.active4(x)
        xs = [x]

        x = self.conv5(x, edge_index)
        x = self.active5(x)
        xs += [x]

        x = self.conv6(x, edge_index)
        x = self.active6(x)
        xs += [x]

        # ~~~~~~~~~~~~Jumping knowledge applied ~~~~~~~~~~~~~~~
        x = self.jk2(xs)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        x = self.lin1(x)
        x = self.active7(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin2(x)

        return x

    def loss(self, pred, label):
        return (torch.sqrt(
            ((pred[:, 0] - label[:, 0]) ** 2).unsqueeze(-1) + ((pred[:, 1] - label[:, 1]) ** 2).unsqueeze(-1) + (
                        (pred[:, 2] - label[:, 2]) ** 2).unsqueeze(-1))).sum()
