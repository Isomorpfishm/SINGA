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
              ligand:Molecule, 
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

    assert ligand is not None, 'Error when loading ligand file'
    assert protein is not None, 'Error when loading protein file'
    assert list_atom_name is not None, 'Missing a list of atom names'
    
    with stderr_redirected(to='obabel.err'):
        (protein_atom_properties_list, protein_atom_pos,
         protein_edge_index, protein_edge_attr) = get_molecular_properties(protein)
        (ligand_atom_properties_list, ligand_atom_pos,
         ligand_edge_index, ligand_edge_attr) = get_molecular_properties(ligand)

    (p_atm_to_l_edge_index, l_to_p_atm_edge_index, p_atm_to_l_edge_attr,
     l_to_p_atm_edge_attr) = get_bonds_protein_ligand(protein, ligand,
                                                      cutoff=cutoff,
                                                      list_atom_name=list_atom_name)

    return (protein_atom_properties_list,  # protein_atoms.x
            ligand_atom_properties_list,  # ligand_atoms.x
            
            protein_atom_pos,
            ligand_atom_pos,

            protein_edge_index,  # protein_atoms <-> protein_atoms
            ligand_edge_index,  # ligand_atoms <-> ligand_atoms
            l_to_p_atm_edge_index,  # ligand_atoms ->  protein_atom
            p_atm_to_l_edge_index,  # protein_atoms -> ligand_atoms

            protein_edge_attr,  # protein_atoms <-> protein_atoms
            ligand_edge_attr,  # ligand_atoms <-> ligand_atoms
            l_to_p_atm_edge_attr,  # ligand_atoms ->  protein_atoms
            p_atm_to_l_edge_attr  # protein_atoms -> ligand_atoms
            )


def create_pyg_graph(protein:Molecule,
                     ligand:Molecule,
                     list_atom_name: list,
                     score:float = 0.0,
                     rmsd:float = 0.0,
                     protein_sasa:float = None,
                     ligand_sasa:float = None,
                     name:str = None,
                     cutoff:float = 4.0) -> pyg.data.HeteroData:
    """
    Create a torch_geometric HeteroGraph of a protein-ligand complex

    Args:
        protein (Molecule):
        ligand (Molecule):
        list_atom_name (list):
        cutoff (float): The maximal distance between two atoms to connect them with an edge. Defaults to 4.0.
        
        score (float, optional): Affinity target. Defaults to None.
        rmsd (float, optional): Used for docking power. Defaults to 0.0.
        name (str, optional): PDB ID for casf output. Defaults to None.
        protein_sasa (float, optional):
        ligand_sasa (float, optional):
   
    Output:
        data (pyg.data.HeteroData): Containing several sets of nodes, and different sets of
                                    edges that can link nodes of the same set or of different sets
    """
    (protein_atm_x,
     ligand_atm_x,
     
     protein_atm_pos,
     ligand_atm_pos,

     protein_atm_to_protein_atm_edge_index,
     ligand_atm_to_ligand_atm_edge_index,
     ligand_atm_to_protein_atm_edge_index,
     protein_atm_to_ligand_atm_edge_index,

     protein_atm_to_protein_atm_edge_attr,
     ligand_atm_to_ligand_atm_edge_attr,
     ligand_atm_to_protein_atm_edge_attr,
     protein_atm_to_ligand_atm_edge_attr
     ) = featurise(protein, ligand, cutoff=cutoff, list_atom_name=list_atom_name)

    data = pyg.data.HeteroData()

    data['protein_atoms'].x = torch.tensor(protein_atm_x)
    data['ligand_atoms'].x = torch.tensor(ligand_atm_x)
    
    data['protein_atoms'].pos = torch.tensor(protein_atm_pos)
    data['ligand_atoms'].pos = torch.tensor(ligand_atm_pos)

    data['protein_atoms', 'linked_to', 'protein_atoms'].edge_index = torch.tensor(
        protein_atm_to_protein_atm_edge_index)
    data['ligand_atoms', 'linked_to', 'ligand_atoms'].edge_index = torch.tensor(
        ligand_atm_to_ligand_atm_edge_index)
    data['ligand_atoms', 'interact_with', 'protein_atoms'].edge_index = torch.tensor(
        ligand_atm_to_protein_atm_edge_index)
    data['protein_atoms', 'interact_with', 'ligand_atoms'].edge_index = torch.tensor(
        protein_atm_to_ligand_atm_edge_index)

    data['protein_atoms', 'linked_to', 'protein_atoms'].edge_attr = torch.tensor(
        protein_atm_to_protein_atm_edge_attr).float()
    data['ligand_atoms', 'linked_to', 'ligand_atoms'].edge_attr = torch.tensor(
        ligand_atm_to_ligand_atm_edge_attr).float()
    data['ligand_atoms', 'interact_with', 'protein_atoms'].edge_attr = torch.tensor(
        ligand_atm_to_protein_atm_edge_attr).float()
    data['protein_atoms', 'interact_with', 'ligand_atoms'].edge_attr = torch.tensor(
        protein_atm_to_ligand_atm_edge_attr).float()

    data.name = name
    #data.rmsd = rmsd
    #data.protein_sasa = protein_sasa
    #data.ligand_sasa = ligand_sasa
    data.y = [score, rmsd, protein_sasa, ligand_sasa]

    return data
    