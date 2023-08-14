import os
import numpy as np
import matplotlib.pyplot as plt
from vanillaGenerate import *

import torch
os.environ['TORCH'] = torch.__version__
print(f"Using Torch version: {torch.__version__}")
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATConv, MLP, GINConv, global_add_pool
from torch_geometric.nn import Linear
import torch_geometric.transforms as T



class GATLayer(nn.Module):
    def __init__(
        self,
        in_features:int,
        out_features:int,
        dropout:float,
        alpha:float,
        concat:bool = True,
    ) -> None:
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        
        # Xavier initialization of weights
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    
    def forward(
        self, 
        input: Tensor, 
        adj: Tensor,
    ) -> Tensor:
        h = torch.mm(input, self.W)
        N = h.size()[0]
        print(N)
        
        # Attention mechanism
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # Masked attention
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj>0, e, zero_vec)
        
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            h_prime
        

class GAT(nn.Module):
    def __init__(
        self,
        dataset,
        hid:int = 8,
        in_head:int = 8,
        out_head:int = 1,
        dropout:float = 0.6,
        concat:bool = False,
        training:bool = True,
    ) -> None:
        super(GAT, self).__init__()
        self.hid = hid
        self.in_head = in_head
        self.out_head = out_head
        self.dropout = dropout
        self.concat = concat
        self.num_features = 7,
        self.num_classes = 2,

        self.conv1 = GATConv(self.num_features, self.hid, heads=self.in_head, dropout=self.dropout)
        self.conv2 = GATConv(self.hid*self.in_head, self.num_classes, concat=self.concat, heads=self.out_head, dropout=self.dropout)
        
    def forward(
        self,
        data:Data,
    ) -> Tensor:
        x, edge_index = data.x, data.edge_index
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)


class Net(nn.Module):
    def __init__(
        self, 
        in_channels:int, 
        hidden_channels:int, 
        out_channels:int, 
        num_layers:int,
        dropout:float = 0.5,
    ) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=dropout)

    def forward(
        self, 
        x:Tensor, 
        edge_index:Tensor, 
        batch:Tensor,
    ):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x)


class Discriminator(nn.Module):
    def __init__(
        self, 
        in_channels:int, 
        hidden_channels:int, 
        out_channels:int, 
        num_layers:int,
        dropout:float = 0.5,
    ) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=dropout)
        self.linear = Linear(out_channels, 1)
        self.activation = nn.Sigmoid()

    def forward(
        self, 
        x:Tensor, 
        edge_index:Tensor, 
        batch:Tensor,
    ):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        
        x = global_add_pool(x, batch)
        x = self.mlp(x)
        x = self.linear(x)
        x = self.activation(x)
        
        return x.view(-1, 1)
