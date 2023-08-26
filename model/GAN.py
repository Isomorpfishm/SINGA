import torch
torch.manual_seed(1)
import torch.nn as nn
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch import Tensor, LongTensor
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv

import tensorflow as tf
from tensorflow import keras
import spektral

import numpy as np
from typing import Any, Optional, Callable, Tuple, Dict, Sequence, NamedTuple



class SINGAModel(keras.Model):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        
        name = kwargs.get('name')
        
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging
        self.vars = {}

    def _build(self):
        raise NotImplementedError
        
    def build(self):
        """ Wraper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        
    def fit(self):
        pass
    
    def predict(self):
        pass






class SINGAModelVAE(SINGAModel):
    def __init__(self, placeholders, num_features:int, num_nodes, features_nonzero, **kwargs):
        super(SINGAModelVAE, self).__init__(**kwargs)
        
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()
        
    def _build(self):
        self.hidden1 = GraphConvolutionalSparse()
        



if __name__ == '__main__':
    

    """

    # Create the Vanilla GNN model
    gat = GAT(data_list[0].num_features, 32, 2)
    print("\n")
    print(gat)

    # Train
    gat.fit(data_list[0], epochs=100)

    # Test
    acc = gat.test(data_list[1])
    print(f'GAT test accuracy: {acc*100:.2f}%')



    a = GetAdjacencyMatrix(Chem.MolFromSmiles(x_smiles[0]))
    x = data_list[0].x
    e = data_list[0].edge_attr
    graph_0 = Graph(x=np.array(x), a=np.array(a), e=np.array(e), y=np.array(0))

    a = GetAdjacencyMatrix(Chem.MolFromSmiles(x_smiles[1]))
    x = data_list[1].x
    e = data_list[1].edge_attr
    graph_1 = Graph(x=np.array(x), a=np.array(a), e=np.array(e), y=np.array(0))

    graph = [[graph_0, None], [graph_1, None]]

    """

