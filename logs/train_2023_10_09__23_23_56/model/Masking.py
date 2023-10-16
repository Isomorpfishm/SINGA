from typing import Dict, List, Tuple
import copy
import random
import numpy as np
random.seed(2023)
np.random.seed(2023)

import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric as pyg
#from torch_geometric.data import HeteroData
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.utils import bipartite_subgraph
from torch_geometric.typing import NodeType



class LigandMasking(nn.Module):
    """
    An nn.Module to mask (ligand) graph's nodes and vertices
    Adapted from Pocket2Mol repository
    """
    def __init__(self,
                 HeteroData:'HeteroData',
                 min_ratio:float=0.1,
                 max_ratio:float=0.9,
                 min_masked:int=1,
                 min_unmasked:int=0,
                 device:str='cuda') -> None:
        super().__init__()
        self.HeteroData = HeteroData
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_masked = min_masked
        self.min_unmasked = min_unmasked
        self.device = device

    def forward(self) -> Tuple[Tensor, Tensor]:
        """
        Forward; Preparing indices for masked and content ligand atoms 

        Args:
            HeteroData (pyg.data.HeteroData): A PyTorch heterogeneous graph

        Output:
            Tuple[Tensor, Tensor]: Tensors of masked and content indices
        """
        masked_ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_atoms = len(self.HeteroData['ligand_atoms']['x'])
        num_atoms_masked = int(num_atoms * masked_ratio)

        if num_atoms_masked < self.min_masked:
            num_atoms_masked = self.min_masked
        if (num_atoms - num_atoms_masked) < self.min_unmasked:
            num_atoms_masked = num_atoms - self.min_unmasked

        # extract masked and content indices for ligands
        idx = np.arange(num_atoms)
        np.random.shuffle(idx)
        idx = torch.LongTensor(idx)
        masked_idx, content_idx = idx[:num_atoms_masked], idx[num_atoms_masked:]
        masked_idx, content_idx = masked_idx.to(self.device), content_idx.to(self.device)

        return masked_idx, content_idx

    def subset_subgraph(self, subset_dict:Dict[NodeType, Tensor]) -> 'HeteroData':
        """
        From torch_geometric.data.hetero_data class method
        
        Returns the induced subgraph containing the node types and
        corresponding nodes in :obj:`subset_dict`.

        If a node type is not a key in :obj:`subset_dict` then all nodes of
        that type remain in the graph.

        Args:
            subset_dict (Dict[str, LongTensor or BoolTensor]): 
                A dictionary holding the nodes to keep for each node type.
                
        Output:
            HeteroGraph (pyg.data.HeteroData): Masked heterogeneous graph
        """
        data = copy.copy(self.HeteroData)
        subset_dict = copy.copy(subset_dict)

        for node_type, subset in subset_dict.items():
            for key, value in self.HeteroData[node_type].items():
                if key == 'num_nodes':
                    if subset.dtype == torch.bool:
                        data[node_type].num_nodes = int(subset.sum())
                    else:
                        data[node_type].num_nodes = subset.size(0)
                elif self.HeteroData[node_type].is_node_attr(key):
                    data[node_type][key] = value[subset]
                else:
                    data[node_type][key] = value

        for edge_type in self.HeteroData.edge_types:
            src, _, dst = edge_type

            src_subset = subset_dict.get(src)
            if src_subset is None:
                src_subset = torch.arange(data[src].num_nodes, device=self.device)
            dst_subset = subset_dict.get(dst)
            if dst_subset is None:
                dst_subset = torch.arange(data[dst].num_nodes, device=self.device)

            edge_index, _, edge_mask = bipartite_subgraph(
                (src_subset, dst_subset),
                self.HeteroData[edge_type].edge_index,
                relabel_nodes=True,
                size=(self.HeteroData[src].num_nodes, self.HeteroData[dst].num_nodes),
                return_edge_mask=True,
            )

            for key, value in self.HeteroData[edge_type].items():
                if key == 'edge_index':
                    data[edge_type].edge_index = edge_index
                elif self.HeteroData[edge_type].is_edge_attr(key):
                    data[edge_type][key] = value[edge_mask]
                else:
                    data[edge_type][key] = value

        return data
