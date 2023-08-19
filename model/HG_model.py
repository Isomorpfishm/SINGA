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
        
    def _common_step(self, 
                     batch:pyg.data.HeteroData, 
                     batch_idx:int, 
                     stage:str) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Common step for trainig, validating and testing
        
        Args:
            batch (pyg.data.HeteroData): A batch
            batch_idx (int): Batch index
            stage (str): train/val
            
        Output:
            Tuple[Tensor, Tensor, Tensor]: (loss, score, real affinities)
        """
        batch_size = batch.y.size(0)
        y_pred = self(batch)
        loss = self.loss_funct(y_pred.view(-1), batch.y.float())
        self.log("Step/{}_loss".format(stage), loss, batch_size=batch_size, sync_dist=True)
        
        return loss, y_pred.view(-1), batch.y
        
    def training_step(self, batch:pyg.data.HeteroData, batch_idx:int) -> Dict:
        """
        Training step
        
        Args:
            batch (pyg.data.HeteroData): A batch
            batch_idx (int): Batch index
            
        Output:
            Dict: (loss, predicted score, real affinity)
        """
        loss, preds, targets = self._common_step(batch, batch_idx, 'train')
        return {'loss':loss, 'train_preds':preds.detach(), 'train_targets':targets.detach()}
        
    def validation_step(self, batch:pyg.data.HeteroData, batch_idx:int) -> Dict:
        """
        Validation step
        
        Args:
            batch (pyg.data.HeteroData): A batch
            batch_idx (int): Batch index
            
        Output:
            Dict: (loss, predicted score, real affinity)
        """
        loss, preds, targets = self._common_step(batch, batch_idx, 'val')
        return {'val_loss':loss, 'val_preds':preds, 'val_targets':targets}
    
    def testing_step(self, batch:pyg.data.HeteroData, batch_idx:int) -> Dict:
        """
        Training step
        
        Args:
            batch (pyg.data.HeteroData): A batch
            batch_idx (int): Batch index
            
        Output:
            Dict: (loss, predicted score, real affinity)
        """
        loss, preds, targets = self._common_step(batch, batch_idx, 'test')
        return {'test_loss':loss, 'test_targes':preds, 'test_pdb_id':targets, 'test_cluster': batch.cluster, 'test_pdb_id': batch.pdb_id}
        
    def common_epoch_end(self, outputs:List, stage:str):
        """
        Called after each epoch (except testing). Calculate CASF metrics
        
        Args:
            outputs (List): Outputs produced by train and val steps
            stage (str): train/val
        """
        loss_name = 'loss' if stage == 'train' else "{}_loss".format(stage)
        loss_batched = torch.stack([x[loss_name] for x in outputs])
        avg_loss = loss_batched.mean()
        all_preds = torch.concat([x["{}_preds".format(stage)] for x in outputs])
        all_targets = torch.concat([x["{}_targets".format(stage)] for x in outputs])
        r2 = tmf.r2_score(all_preds, all_targets)
        pearson = tmf.pearson_corrcoef(all_preds, all_targets)
        metrics_dict = {
            "ep_end_{}/loss".format(stage): avg_loss,
            "ep_end_[}/r2_score".format(stage): r2,
            "ep_end_{}/pearson".format(stage): pearson,
        }
        self.log_dict(metrics_dict, sync_dist=True)
        
    def training_epoch_end(self, outputs:List):
        self.common_epoch_end(outputs, 'training')
                
    def validation_epoch_end(self, outputs:List):
        self.common_epoch_end(outputs, 'val')

    def configure_optimiser(self):
        optimiser = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {"optimiser": optimiser}
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
        
    def predict(self, data:pyg.data.Batch) -> float:
        """
        Use the model to predict scores
        
        Args:
            data (pyg.data.Batch): Data
            
        Output:
            float: Predicted scores
        """
        self.eval()
        with torch.no_grad():
            score = self(data)
        return score[0].item()
        