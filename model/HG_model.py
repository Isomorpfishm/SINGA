import os
import sys
import logging
import warnings
from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F
import torch_geometric as pyg
import pytorch_lightning as pl
import torchmetrics.functional as tmf

try:
    from HG_embedding import HG_Net
except:
    from model.HG_embedding import HG_Net


### Adapted from: https://github.com/KevinCrp/HGScore/blob/main/HGScore/model.py ###

class HG_Model(pl.LightningModule):
    """ A PyTorch Lightning model """
    def __init__(self,
                 hidden_channels_pa:Union[int, List[int]],
                 hidden_channels_la:Union[int, List[int]],
                 num_layers:int,
                 dropout:float,
                 heads:int,
                 hetero_aggr:str,
                 mlp_channels:List,
                 lr:float,
                 weight_decay:float,
                 molecular_embedding_size:int,
                 str_for_hparams:str='') -> None:
        """ 
        Args:
            hidden_channels_pa (Union[int, List[int]]): Size of channels for protein
            hidden_channels_la (Union[int, List[int]]): Size of channels for ligand
            num_layers (int): Number of layers
            dropout (float): Dropout rate
            heads (int): Number of heads
            hetero_aggr (str): How the hetero aggregation is performed
            mlp_channels (List): List of final MLP channels size
            lr (float): Learning rate
            weight_decay (float): Weight decay
            molecular_embedding_size (int): Number of timesteps for molecular embedding
            str_for_hparams (str[Optional]): Allowing to save supplementary information for Tesorboard 
        """
        super().__init__()
        self.save_hyperparameters()
        if isinstance(hidden_channels_pa, int):
            hidden_channels_pa = num_layers * [hidden_channels_pa]
        if isinstance(hidden_channels_la, int):
            hidden_channels_la = num_layers * [hidden_channels_la]
        if len(hidden_channels_pa) != num_layers:
            logging.error("Num_layer does not match the given layer size")
            sys.exit()
        
        self.model = HG_Net(list_hidden_channels_pa=hidden_channels_pa,
                            list_hidden_channels_la=hidden_channels_la,
                            num_layers=num_layers,
                            hetero_aggr=hetero_aggr,
                            mlp_channels=mlp_channels,
                            molecular_embedding_size=molecular_embedding_size,
                            dropout=dropout,
                            heads=heads)
        self.loss_funct = F.mse_loss
        self.lr = lr
        self.weight_decay = weight_decay

    def get_nb_parameters(self, only_trainable:bool=False):
        """
        Get the number of model's parameters
        
        Args:
            only_trainable (bool[Optional]): Consider only trainable parameters
            
        Output:
            int: Number of parameters
        """
        return self.model.get_nb_parameters(only_trainable)
        
    def forward(self, data:pyg.data.HeteroData) -> Tensor:
        """
        Forward
        
        Args:
            data (pyg.data.HeteroData): A Hetero batch
        
        Output:
            Tensor: Scores 
        """
        return self.model(data.x_dict, data.edge_index_dict, data.edge_attr_dict, data.batch_dict)
        
