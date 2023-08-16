from typing import Dict, List, Tuple

import torch
import torch_geometric as pyg
from torch_geometric.nn import HeteroConv

try:
    from layers import (AFP_GATE_GRUConv_InterMol,
                        AFP_GATE_GRUConv_IntraMol,
                        ADP_GATGRUConv_InterMol,
                        AFP_GETGRUConv_IntraMol,
                        AFP_GATGRUConvMol,
                        molecular_pooling)
except:
    from model.layers import (AFP_GATE_GRUConv_InterMol,
                              AFP_GATE_GRUConv_IntraMol,
                              ADP_GATGRUConv_InterMol,
                              AFP_GETGRUConv_IntraMol,
                              AFP_GATGRUConvMol,
                              molecular_pooling)
                              
### Adapted from https://github.com/KevinCrp/HGScore/blob/main/HGScore/networks/heterogeneous_afp.py
NB_ATOM_FTS = 23
NB_INTRA_BOND_FTS = 6
NB_INTER_BOND_FTS = 6


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
                                                                                         hidden_channels_pa,
                                                                                         dropout=dropout,
                                                                                         edge_dim=NB_INTRA_BOND_FTS)
                                                                                         
    conv_dict[('ligand_atoms', 'interact_with', 'protein_atoms')] = AFP_GATE_GRUConv_InterMol(NB_ATOM_FTS,
                                                                                              hidden_channels_pa,
                                                                                              dropout=dropout,
                                                                                              edge_dim=NB_INTRA_BOND_FTS)
                                                                                           
    conv_dict[('protein_atoms', 'interact_with', 'ligand_atoms')] = AFP_GATE_GRUConv_InterMol(NB_ATOM_FTS,
                                                                                              hidden_channels_pa,
                                                                                              dropout=dropout,
                                                                                              edge_dim=NB_INTRA_BOND_FTS)

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
        conv_dict[('ligand_atoms', 'interact_with', 'protein_atoms')] = AFP_GATGRUConv_InterMol(in_channels_la, hidden_channels_pa, hidden_channels_pa, edge_dim=None, heads=heads, dropout=dropout)
        conv_dict[('protein_atoms', 'interact_with', 'ligand_atoms')] = AFP_GATGRUConv_InterMol(in_channels_pa, hidden_channels_la, hidden_channels_la, edge_dim=None, heads=heads, dropout=dropout)
        list_dico.append(conv_dict)
        
    return list_dico

