from typing import Tuple, List
import random
import numpy as np
np.random.seed(2023)

import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric as pyg
#from torch_geometric.data import HeteroData
from torch_geometric.utils.subgraph import subgraph


class LigandMasking(nn.Module):
    """
    An nn.Module to mask (ligand) graph's nodes and vertices
    Adapted from Pocket2Mol repository
    """
    def __init__(self,
                 min_ratio:float=0.1,
                 max_ratio:float=0.9,
                 min_masked:int=1,
                 min_unmasked:int=0,
                 device:str='cuda') -> None:
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_masked = min_masked
        self.min_unmasked = min_unmasked
        self.device = device

    def forward(self, HeteroData):
        """
        Forward

        Args:
            HeteroData (pyg.data.HeteroData): A PyTorch heterogeneous graph

        Output:
            Masked graphs
        """
        masked_ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_atoms = len(HeteroData['ligand_atoms']['x'])
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

        # change ligand atom attributes and coordinates
        HeteroData['ligand_atoms']['masked_x'] = HeteroData['ligand_atoms']['x'][masked_idx]
        HeteroData['ligand_atoms']['masked_pos'] = HeteroData['ligand_atoms']['pos'][masked_idx]

        HeteroData['ligand_atoms']['content_x'] = HeteroData['ligand_atoms']['x'][content_idx]
        HeteroData['ligand_atoms']['content_pos'] = HeteroData['ligand_atoms']['pos'][content_idx]

        # change ligand bond indices and attributes
        if HeteroData[('ligand_atoms', 'linked_to', 'ligand_atoms')]['edge_index'].size(1) != 0:
            HeteroData[('ligand_atoms', 'linked_to', 'ligand_atoms')]['content_edge_index'], \
            HeteroData[('ligand_atoms', 'linked_to', 'ligand_atoms')]['content_edge_attr'] = subgraph(content_idx,
                                                                                                      HeteroData[('ligand_atoms', 'linked_to', 'ligand_atoms')]['edge_index'],
                                                                                                      edge_attr=HeteroData[('ligand_atoms', 'linked_to', 'ligand_atoms')]['edge_attr'],
                                                                                                      relabel_nodes=True)
        else:
            HeteroData[('ligand_atoms', 'linked_to', 'ligand_atoms')]['content_edge_index'] = torch.empty([2, 0], dtype=torch.long)
            HeteroData[('ligand_atoms', 'linked_to', 'ligand_atoms')]['content_edge_attr'] = torch.empty([0], dtype=torch.long)

        #subset_dict = {'ligand_atoms': content_idx}
        #HeteroData = HeteroData.subgraph(subset_dict)

        return HeteroData, masked_idx, content_idx
