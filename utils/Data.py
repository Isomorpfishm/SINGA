import os
import re
import random
import multiprocessing as mp
from typing import List, Tuple, Union, Dict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch_geometric as pyg
#from torch_geometric.data import Dataset
import pytorch_lightning as pl

import oddt.interactions as interactions
from oddt.spatial import distance
from oddt.toolkits.ob import Molecule, readfile
from openbabel import openbabel


### Adapted from https://github.com/KevinCrp/HGScore/blob/main/HGScore/featurizer.py
###              https://github.com/KevinCrp/HGScore/blob/main/HGScore/data.py

def atom_type_one_hot(atomic_num:int) -> List[int]:
    """
    43 + 1 atom types considered:
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
         'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
         'Be', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li',
         'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt',
         'Hg', 'Pb', 'Ga', 'Unknown']
    
    Args:
        atomic_num (int): Atom's atomic number
        
    Output:
        List[int]: One-hot vector
    """
    one_hot = 44 * [0]
    used_atom_num = [ 3,  4,  5,  6,  7,  8,  9, 11, 12, 13,
                     14, 15, 16, 17, 19, 20, 22, 23, 24, 25,
                     26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 
                     40, 46, 47, 48, 49, 50, 51, 53, 78, 79,
                     80, 81, 82]
    d_atom_num = {3:0, 4:1, 5:2, 6:3, 7:4, 8:5, 9:6, 11:7, 12:8, 13:9,
                  14:10, 15:11, 16:12, 17:13, 19:14, 20:15, 22:16, 23:17, 24:18, 25:19,
                  26:20, 27:21, 28:22, 29:23, 30:24, 31:25, 32:26, 33:27, 34:28, 35:29,
                  40:30, 46:31, 47:32, 48:33, 49:34, 50:35, 51:36, 53:37, 78:38, 79:39,
                  80:40, 81:41, 82:42}
    idx = 43
    if atomic_num in used_atom_num:
        idx = d_atom_num[atomic_num]
    one_hot[idx] = 1
    
    return one_hot


def atom_hybridisation_one_hot(hybridisation:int) -> List[int]:
    """
    Hybridisation considered:
        [other, sp, sp2, sp3, sq.planer, trig, bipy, octahedral]
        
    Args:
        hybridisation (int): Hybridisation
        
    Output:
        List[int]: One-hot vector
    """
    one_hot_hybridisation = 7 * [0]
    if hybridisation not in [1, 2, 3, 4, 5, 6]:
        hybridisation = 0
    one_hot_hybridisation[hybridisation] = 1
    
    return one_hot_hybridisation


def atom_degree_one_hot(degree:int) -> List[int]:
    """
    Hetero/heavy degree considered: 
        [0, 1, 2, 3, 4, 5, 6+]
    
    Args:
        degree (int): Hetero/heavy degree
        
    Output:
        List[int]: One-hot vector
    """
    oh_degree = 7 * [0]
    if degree > 6:
        oh_degree[6] = 1
    else:
        oh_degree[degree] = 1
        
    return oh_degree


def get_bond_properties(bond:openbabel.OBBond) -> List:
    """
    Bond properties considered:
        - Bond length
        - Is in an aromatic ring
        - Is in a ring
        - Is single bond
        - Is double bond
        - Is triple bond
        
    Args:
        bond (openbabel.OBBond): openbabel bond
        
    Output:
        List: A list of bond properties
    """
    order = bond.GetBondOrder()
    length = bond.GetLength()
    aromatic = bond.IsAromatic()
    ring = bond.IsInRing()
    
    return [length, aromatic, ring, order==1, order==2, order==3]
    
    
def get_molecular_properties(mol:Molecule) -> Tuple[List, List, List]:
    """
    Compute all atomic and molecular properties of a molecule
    
    Args:
        pybel_mol (Molecule): An ODDT Molecule object
        
    Output:
        Tuple[List, List, List]: (Atoms' properties, edge index, edge attributes)
    """
    oh_atom_type = np.array(list(map(atom_type_one_hot, mol.atom_dict['atomicnum'].tolist())))
    oh_hybridisation = np.array(list(map(atom_hybridisation_one_hot, mol.atom_dict['hybridization'].tolist())))
    partial_charge = mol.atom_dict['charge'].reshape(-1, 1)
    hydrophobic = mol.atom_dict['ishydrophobe'].reshape(-1, 1)
    isaromatic = mol.atom_dict['isaromatic'].reshape(-1, 1)
    isacceptor = mol.atom_dict['isacceptor'].reshape(-1, 1)
    isdonor = mol.atom_dict['isdonor'].reshape(-1, 1)
    isdonorh = mol.atom_dict['isdonor'].reshape(-1, 1)
    isminus = mol.atom_dict['isminus'].reshape(-1, 1)
    isplus = mol.atom_dict['isplus'].reshape(-1, 1)
    
    atom_properties_list = np.concatenate((oh_atom_type, oh_hybridisation, partial_charge, hydrophobic, isaromatic, isacceptor, isdonor, isdonorh, isminus, isplus), axis=1).tolist()
    
    edge_index, edge_attr = [[], []], []
    
    for bond in mol.bonds:
        ob_bond = bond.OBBond
        begin_id = ob_bond.GetBeginAtom().GetIdx() - 1
        end_id = ob_bond.GetEndAtom().GetIdx() - 1
        edge_index[0] += [begin_id, end_id]
        edge_index[1] += [end_id, begin_id]
        edge_attr += [get_bond_properties(ob_bond), get_bond_properties(ob_bond)]
        
    return (atom_properties_list, edge_index, edge_attr)


class CrossdockedDataSet(Dataset):
    # PyTorch Geometric Dataset module 
    def __init__(self, list_IDs:List) -> None:
        super().__init__()
        self.list_IDs = list_IDs
        
    def __getitem__(self, index) -> Tensor:
        path_to_pt = self.list_IDs[index]
        X = torch.load(path_to_pt)
        X = X.to(torch.device('cuda:0'))
        return X
    
    def __len__(self):
        return len(self.list_IDs)


@dataclass
class CrossdockedDataModule(pl.LightningDataModule):
    # PyTorch Lightning Data Module
    root:str                                             # path to data directory
    index:str                                            # path to split index
    split_ratio:float = field(default=0.9)               # ratio of samples for training from split_by_name.pt
    atomic_distance_cutoff:float = field(default=4.0)    # cutoff for interatomic distance
    batch_size:int = field(default=1)                    # batch size
    num_workers:int = field(default=1)                   # number of workers
    persistent_workers:bool = field(default=True)        # use persistant workers in dataloader

    def __post_init__(self):
        super().__init__()
        
    def prepare_data(self):
        pass
        
    def setup(self):
        split_idx = torch.load(self.index)
        
        lt_train_ori, lt_test = [], []
        for i in range(len(split_idx['train'])):
            filePth = os.path.join(self.root, str(split_idx['train'][i][0].split(".")[0]) + ".pt")
            fileExist = os.path.isfile(filePth)
            if fileExist:
                lt_train_ori.append(filePth)
            else:
                continue
        for i in range(len(split_idx['test'])):
            filePth = os.path.join(self.root, str(split_idx['test'][i][0].split(".")[0]) + ".pt")
            fileExist = os.path.isfile(filePth)
            if fileExist:
                lt_test.append(filePth)
            else:
                continue
        
        lt_train = lt_train_ori[:int(len(lt_train_ori)*self.split_ratio)]
        lt_val = lt_train_ori[int(len(lt_train_ori)*self.split_ratio):]
        
        """
        for i in lt_train[:5000]:
            try:
                self.dt_train.append(torch.load(i).to('cuda'))
            except FileNotFoundError:
                continue
        
        for i in lt_val:
            try:
                self.dt_val.append(torch.load(i).to('cuda'))
            except FileNotFoundError:
                continue
        #self.dt_test = [torch.load(i).to('cuda') for i in lt_test]
        """
        self.dt_train = CrossdockedDataSet(lt_train[5000:10000])
        self.dt_val = CrossdockedDataSet(lt_val)
        
        
    def train_dataloader(self):
        return pyg.loader.DataLoader(dataset=self.dt_train,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=self.num_workers,
                                     persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return pyg.loader.DataLoader(dataset=self.dt_val,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=self.num_workers,
                                     persistent_workers=self.persistent_workers)
    
    def test_dataloader(self):
        return pyg.loader.DataLoader(dataset=self.dt_test,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=self.num_workers,
                                     persistent_workers=self.persistent_workers)
       
