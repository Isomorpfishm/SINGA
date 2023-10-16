import os
import re
import multiprocessing as mp
from typing import Tuple, Dict, Union, List
from dataclasses import dataclass, field

from oddt.toolkits.ob import Molecule
import biopandas.pdb as bpdb
import torch
import torch_geometric as pyg

try:
    from Data import get_molecular_properties
    from PLInteraction import get_bonds_protein_ligand
    from redirect import stderr_redirected
except:
    from utils.Data import get_molecular_properties
    from utils.PLInteraction import get_bonds_protein_ligand
    from utils.redirect import stderr_redirected


### Adapted from https://github.com/KevinCrp/HGScore/blob/main/HGScore/featurizer.py ###

def featurise(protein:Molecule, 
              list_atom_name:list,
              cutoff:float) -> Tuple:
    """
    Featurise a protein and a ligand to a set of nodes and edges

    Args:
        protein (ob.Molecule):
        ligand (ob.Molecule): 
        cutoff (float): 

    Output:
        Tuple: Nodes and edges
    """

    assert protein is not None, 'Error when loading protein file'

    with stderr_redirected(to='obabel.err'):
        (protein_atom_properties_list, protein_atom_pos,
         protein_edge_index, protein_edge_attr) = get_molecular_properties(protein)

    return (protein_atom_properties_list,  # protein_atoms.x
            protein_atom_pos,
            protein_edge_index,  # protein_atoms <-> protein_atoms
            protein_edge_attr,  # protein_atoms <-> protein_atoms
            )


def create_pyg_graph(protein:Molecule,
                     list_atom_name:list,
                     name:str = None,
                     cutoff:float = 4.0) -> pyg.data.HeteroData:
    """
    Create a torch_geometric HeteroGraph of a protein-ligand complex

    Args:
        protein (Molecule):
        list_atom_name (list):
        cutoff (float): The maximal distance between two atoms to connect them with an edge. Defaults to 4.0.
        name (str, optional): PDB ID for casf output. Defaults to None.
   
    Output:
        data (pyg.data.HeteroData)
    """
    (protein_atm_x,
     protein_atm_pos,
     protein_atm_to_protein_atm_edge_index,
     protein_atm_to_protein_atm_edge_attr,
     ) = featurise(protein, cutoff=cutoff, list_atom_name=list_atom_name)

    data = pyg.data.HeteroData()

    data['protein_atoms'].x = torch.tensor(protein_atm_x)
    data['protein_atoms'].pos = torch.tensor(protein_atm_pos)
    data['protein_atoms', 'linked_to', 'protein_atoms'].edge_index = torch.tensor(
        protein_atm_to_protein_atm_edge_index
    )
    data['protein_atoms', 'linked_to', 'protein_atoms'].edge_attr = torch.tensor(
        protein_atm_to_protein_atm_edge_attr
    ).float()

    
    atomic_dict = {}
    atomic_numbers = protein.atom_dict['atomicnum'].tolist()
    atomic_numbers = torch.tensor(atomic_numbers).long()
    atomic_dict['protein_atoms'] = atomic_numbers
    
    data.name = name
    data.atomicnum = atomic_dict

    return data
    