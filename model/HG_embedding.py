from typing import Dict, List, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
import torch_geometric as pyg
from torch_geometric.nn import HeteroConv
from torch_geometric.nn.models import MLP

try:
    from HG_layers import (AFP_GATE_GRUConv_InterMol,
                           AFP_GATE_GRUConv_IntraMol,
                           AFP_GATGRUConv_InterMol,
                           AFP_GATGRUConv_IntraMol,
                           AFP_GATGRUConvMol,
                           molecular_pooling)
except:
    from model.HG_layers import (AFP_GATE_GRUConv_InterMol,
                                 AFP_GATE_GRUConv_IntraMol,
                                 AFP_GATGRUConv_InterMol,
                                 AFP_GATGRUConv_IntraMol,
                                 AFP_GATGRUConvMol,
                                 molecular_pooling)
                              
### Adapted from https://github.com/KevinCrp/HGScore/blob/main/HGScore/networks/heterogeneous_afp.py
NB_ATOM_FTS = 59
NB_INTRA_BOND_FTS = 6
NB_INTER_BOND_FTS = 11


def get_het_conv_first_layer(hidden_channels_pa:int,
                             hidden_channels_la:int,
                             dropout:float = 0.0) -> Dict:
    """
    Produce a dictionary for heterogeneous graph convolution
    Used to create the first layer of the HetGNN network
    
    Args:
        hidden_channels_pa (int): Input channel size for the protein
        hidden_channels_la (int): Input channel size for the ligand
        dropout (float [Optional]): Dropout rate. Default is 0.0
        
    Output:
        Dict: Dictionary containing the layer
    """
    conv_dict = {}
    
    conv_dict[('protein_atoms', 'linked_to', 'protein_atoms')] = AFP_GATE_GRUConv_IntraMol(NB_ATOM_FTS,
                                                                                           hidden_channels_pa,
                                                                                           dropout=dropout,
                                                                                           edge_dim=NB_INTRA_BOND_FTS)
                                                                                           
    conv_dict[('ligand_atoms', 'linked_to', 'ligand_atoms')] = AFP_GATE_GRUConv_IntraMol(NB_ATOM_FTS,
                                                                                         hidden_channels_la,
                                                                                         dropout=dropout,
                                                                                         edge_dim=NB_INTRA_BOND_FTS)
                                                                                         
    conv_dict[('ligand_atoms', 'interact_with', 'protein_atoms')] = AFP_GATE_GRUConv_InterMol((NB_ATOM_FTS, NB_ATOM_FTS),
                                                                                              hidden_channels_pa,
                                                                                              dropout=dropout,
                                                                                              edge_dim=NB_INTER_BOND_FTS)
                                                                                           
    conv_dict[('protein_atoms', 'interact_with', 'ligand_atoms')] = AFP_GATE_GRUConv_InterMol((NB_ATOM_FTS, NB_ATOM_FTS),
                                                                                              hidden_channels_la,
                                                                                              dropout=dropout,
                                                                                              edge_dim=NB_INTER_BOND_FTS)

    return conv_dict


def get_het_conv_layer(list_hidden_channels_pa:List[int],
                       list_hidden_channels_la:List[int],
                       heads:int = 1,
                       dropout:float = 0.0) -> Dict:
    """ 
    Produce a dictionary for heterogeneous graph convolution
    Used to create the following layers of HetGNN
    
    Args:
        list_hidden_channels_pa (List[int]): Channel size for proteins
        list_hidden_channels_la (List[int]): Channel size for ligands
        heads (int, [Optional): Number of heads. Default = 1
        dropout (float, [Optional]): Dropout rate. Default = 0.0
    """
    list_dico = []
    
    for in_channels_pa, in_channels_la, hidden_channels_pa, hidden_channels_la in zip(list_hidden_channels_pa[:-1], list_hidden_channels_la[:-1], list_hidden_channels_pa[1:], list_hidden_channels_la[1:]):
        conv_dict = {}
        conv_dict[('protein_atoms', 'linked_to', 'protein_atoms')]    = AFP_GATGRUConv_IntraMol(in_channels_pa, hidden_channels_pa, hidden_channels_pa, edge_dim=None, heads=heads, dropout=dropout)
        conv_dict[('ligand_atoms', 'linked_to', 'ligand_atoms')]      = AFP_GATGRUConv_IntraMol(in_channels_la, hidden_channels_la, hidden_channels_la, edge_dim=None, heads=heads, dropout=dropout)
        conv_dict[('ligand_atoms', 'interact_with', 'protein_atoms')] = AFP_GATGRUConv_InterMol((in_channels_la, in_channels_pa), 
                                                                                                hidden_channels_pa, 
                                                                                                hidden_channels_pa, 
                                                                                                edge_dim=None, heads=heads, dropout=dropout)
        conv_dict[('protein_atoms', 'interact_with', 'ligand_atoms')] = AFP_GATGRUConv_InterMol((in_channels_pa, in_channels_la), 
                                                                                                hidden_channels_la, 
                                                                                                hidden_channels_la, 
                                                                                                edge_dim=None, heads=heads, dropout=dropout)
        list_dico.append(conv_dict)
        
    return list_dico


class HeteroAFP_Atomic(nn.Module):
    """ Embedding layers for atoms """
    def __init__(self,
                 list_hidden_channels_pa:List[int],
                 list_hidden_channels_la:List[int],
                 num_layers:int,
                 dropout:float,
                 heads:int,
                 hetero_aggr:str = 'sum'):
        """
        Construct the embedding layers for atoms

        Args:
            list_hidden_channels_pa (List[int]): Size of the channels for protein
            list_hidden_channels_la (List[int]): Size of the channels for ligand
            num_layers (int): Number of layers
            dropout (float): Dropout rate
            heads (int): Number of heads
            hetero_aggr (str): Mode for heterogeneous aggregation
        """
        super().__init__()
        first_layer_dict = get_het_conv_first_layer(list_hidden_channels_pa[0], list_hidden_channels_la[0], dropout)
        list_other_layer_dict = get_het_conv_layer(list_hidden_channels_pa, list_hidden_channels_la, heads, dropout)
        self.conv_list = nn.ModuleList( [HeteroConv(first_layer_dict, aggr=hetero_aggr)] )
        self.num_layers = num_layers

        for i in range(num_layers-1):
            self.conv_list.append(HeteroConv(list_other_layer_dict[i], aggr=hetero_aggr))

    def forward(self, x:Dict, edge_index:Dict, edge_attr:Dict) -> Dict:
        """
        Forward

        Args:
            x (Dict): Dictionary of node features
            edge_index (Dict): Dictionary of edge_index
            edge_attr (Dict): Dictionary of edge attributes

        Output:
            Embedded atoms
        """
        x_dict = self.conv_list[0](x, edge_index, edge_attr)
        x = {key: x.relu() for key, x in x_dict.items()}
        edge_attr = {key: edge_attr[key] for key in [('ligand_atoms', 'interact_with', 'protein_atoms'), ('protein_atoms', 'interact_with', 'ligand_atoms')]}
        for conv in self.conv_list[1:]:
            x_dict = conv(x, edge_index)
            x = {key: x.relu() for key, x in x_dict.items()}
        return x


class AFP_Hetero_Molecular(nn.Module):
    """ Embedding layers for molecules """
    def __init__(self, 
                 hidden_channels_pa:int,
                 hidden_channels_la:int,
                 out_channels_pa:int,
                 out_channels_la:int,
                 molecular_embedding_size:int,
                 dropout:float,
                 heads:int):
        """
        Construct the embedding layers for molecules
        
        Args:
            hidden_channels_pa (int): Size of the channels for protein
            hidden_channels_la (int): Size of the channels for ligand
            out_channels_pa (int): Size of the output channels for protein
            out_channels_la (int): Size of the output channels for ligand
            molecular_embedding_size (int): Number of timestep for molecular embedding
            dropout (float): Dropout rate
            heads (int): Number of heads
        """
        super().__init__()
        self.gcn_pa = self.gcn_la = None
        self.lin_pa = self.lin_la = None
        self.molecular_embedding_size = molecular_embedding_size

        self.gcn_pa = AFP_GATGRUConvMol(hidden_channels_pa, hidden_channels_pa, hidden_channels_pa, dropout, None, heads)
        self.lin_pa = Linear(hidden_channels_pa, out_channels_pa)  
        
        self.gcn_la = AFP_GATGRUConvMol(hidden_channels_la, hidden_channels_la, hidden_channels_la, dropout, None, heads)
        self.lin_la = Linear(hidden_channels_la, out_channels_la)

    def forward(self, x_dict:Dict, edge_index_dict:Dict) -> Tuple[Tensor, Tensor]:
        """
        Forward

        Args:
            x_dict (Dict): Dictionary of node features
            edge_index_dict (Dict): Dictionary of edge_index

        Output:
            Tuple[Tensor, Tensor]: Molecular embedding of (protein, ligand)
        """
        for _ in range(self.molecular_embedding_size):
            x_dict['pa_embedding'] = self.gcn_pa(x_dict['protein_atoms'], x_dict['pa_embedding'], edge_index_dict['pa_embedding'])
            x_dict['la_embedding'] = self.gcn_la(x_dict['ligand_atoms'], x_dict['la_embedding'], edge_index_dict['la_embedding'])

        y_pa = self.lin_pa(x_dict['pa_embedding'])
        y_la = self.lin_la(x_dict['la_embedding'])
        
        return y_pa, y_la


class HG_Net(nn.Module):
    def __init__(self,
                 list_hidden_channels_pa:List[int],
                 list_hidden_channels_la:List[int],
                 num_layers:int,
                 hetero_aggr:str,
                 mlp_channels:List[int],
                 molecular_embedding_size:int,
                 dropout:float,
                 heads:int):
        """
        Construct the model

        Args:
            list_hidden_channels_pa (List[int]):
            list_hidden_channels_la (List[int]):
            num_layers (int):
            hetero_aggr (str):
            mlp_channels (List[int]):
            molecular_embedding_size (int):
            dropout (float):
            heads (int):
        """
        super().__init__()
        self.gcn_atm = HeteroAFP_Atomic(
            list_hidden_channels_pa=list_hidden_channels_pa,
            list_hidden_channels_la=list_hidden_channels_la,
            num_layers=num_layers,
            dropout=dropout,
            heads=heads,
            hetero_aggr=hetero_aggr,
            )
        self.gcn_mol = AFP_Hetero_Molecular(
            hidden_channels_pa=list_hidden_channels_pa[-1],
            hidden_channels_la=list_hidden_channels_la[-1],
            out_channels_pa=list_hidden_channels_pa[-1],
            out_channels_la=list_hidden_channels_la[-1],
            molecular_embedding_size=molecular_embedding_size,
            dropout=dropout,
            heads=heads,
            )
        self.mlp = MLP(channel_list=mlp_channels, dropout=dropout)

    def forward(self, x:Dict, edge_index:Dict, edge_attr:Dict, batch:Dict) -> Tensor:
        """
        Forward

        Args:
            x (Dict): Dictionary of atomic node feature
            edge_index (Dict): Dictionary of edge indices
            edge_attr (Dict): Dictionary of edge attributes
            batch (Dict): Dictionary of batches

        Output:
            Tensor: Score
        """
        y_dict = self.gcn_atm(x, edge_index, edge_attr)
        mol_x_dict, mol_edge_index_dict = molecular_pooling(y_dict, edge_index, batch)
        x_pa, x_la = self.gcn_mol(mol_x_dict, mol_edge_index_dict)
        x_fp = torch.cat((x_pa, x_la), axis=1)
        x_fp = x_fp.unsqueeze(-1)
        return self.mlp(x_fp).transpose(1, -1)
        

    def get_nb_parameters(self, only_trainable:bool=False) -> int:
        """
        Get the number of network's parameters

        Args:
            only_trainable (bool [Optional]): Consider only trainable parameters. 

        Output:
            int: Number of parameters
        """
        nb = 0
        
        if only_trainable:
            nb += sum(p.numel() for p in self.gcn_atm.parameters() if p.requires_grad)
            nb += sum(p.numel() for p in self.gcn_mol.parameters() if p.requires_grad)
            nb += sum(p.numel() for p in self.mlp.parameters() if p.requires_grad)
            return nb
        
        nb += sum(p.numel() for p in self.gcn_atm.parameters())
        nb += sum(p.numel() for p in self.gcn_mol.parameters())
        nb += sum(p.numel() for p in self.mlp.parameters())
        return nb
