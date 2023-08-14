from typing import Optional
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json

import freesasa
import numpy as np
import networkx as nc

from rdkit import Chem
from rdkit.Chem import rdFreeSASA
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

import tensorflow as tf
from tensorflow import keras
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx




def accuracy(y_pred, y_true):
    """Calculate accuracy."""
    return torch.sum(y_pred==y_true)/len(y_true)


def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x==s, permitted_list))]

    return binary_encoding


def get_atom_features(atom, use_chirality=True, hydrogens_implicit=True):
    permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
                               'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
                               'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li',
                               'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt',
                               'Hg', 'Pb', 'Unknown']

    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), 
                                             [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), 
                                         [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridization_type_enc = one_hot_encoding(str(atom.GetHybridization()), 
                                              ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER'])
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    atomic_mass_scaled = [float((atom.GetMass()-10.812)/116.092)]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())-1.5)/0.6)]
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())-0.64)/0.76)]

    atom_feature_vector = atom_type_enc + \
                          n_heavy_neighbors_enc + \
                          formal_charge_enc + \
                          hybridization_type_enc + \
                          is_in_a_ring_enc + \
                          is_aromatic_enc + \
                          atomic_mass_scaled + \
                          vdw_radius_scaled + \
                          covalent_radius_scaled

    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), 
                                              ["CHI_UNSPECIFIED", 
                                               "CHI_TETRAHEDRAL_CW", 
                                               "CHI_TETRAHEDRAL_CCW", 
                                               "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc

    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond, use_stereochemistry=True):
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, 
                                    Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE, 
                                    Chem.rdchem.BondType.AROMATIC]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)
    
    
def RetrieveMolecularFeatures(molList:list, labelList:list):
    """
    Inputs:

    molList   = [smiles_1, smiles_2, ....] ... a list of mol files
    labelList = [y_1, y_2, ...]            ... a list of numerial labels for the SMILES strings 
                                               (such as associated pKi values)

    Outputs:

    data_list = [G_1, G_2, ...] ... 
    a list of torch_geometric.data.Data objects which represent labeled molecular graphs 
    that can readily be used for machine learning
    """

    data_list = []

    for (mol, y_val) in zip(molList, labelList):
        # get feature dimen sions
        try:
            n_nodes = mol.GetNumAtoms()
            n_edges = 2 * mol.GetNumBonds()
            unrelated_smiles = "O=O"
            unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
            n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
            n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))
        except AttributeError:
            print("Extraction of molecular features failed!")
            continue

        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))
        pos = list()
        
        for idx, atom in enumerate(mol.GetAtoms()):
            X[atom.GetIdx(), :] = get_atom_features(atom)
            coords = mol.GetConformer().GetAtomPosition(idx)
            pos.append([coords.x, coords.y, coords.z])

        X = torch.tensor(X, dtype = torch.float)
        pos = torch.tensor(np.array(pos), dtype = torch.float)

        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)

        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))

        for (k, (i, j)) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))

        EF = torch.tensor(EF, dtype=torch.float)

        # construct label tensor
        y_tensor = torch.tensor(np.array([y_val]), dtype=torch.float)

        # construct Pytorch Geometric data object and append to data list
        data_list.append(Data(x=X, edge_index=E, edge_attr=EF, y=y_tensor, pos=pos))

    return data_list


def ClassifyAtoms(path, mol, cc_index, polar_atoms=[7, 8, 15, 16]):
    #Taken from https://github.com/mittinatten/freesasa/blob/master/src/classifier.c
    with open(path, 'r') as file:
        symbol_radius = json.load(file)
    """
    symbol_radius = {"H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80, "S": 1.80, "SE": 1.90, "ZN": 1.39, \
    "F": 1.47, "CL": 1.75, "BR": 1.83, "I": 1.98, \
    "LI": 1.81, "BE": 1.53, "B": 1.92, \
    "NA": 2.27, "MG": 1.74, "AL": 1.84, "SI": 2.10, \
    "K": 2.75, "CA": 2.31, "GA": 1.87, "GE": 2.11, "AS": 1.85, \
    "RB": 3.03, "SR": 2.49, "IN": 1.93, "SN": 2.17, "SB": 2.06, "TE": 2.06}
    """

    radii = [] 
    for i, atom in enumerate(mol.GetAtoms()):
        if i not in cc_index:
            continue
        atom.SetProp("SASAClassName", "Apolar") # mark everything as apolar to start
        if atom.GetAtomicNum() in polar_atoms: #identify polar atoms and change their marking
            atom.SetProp("SASAClassName", "Polar") # mark as polar
        elif atom.GetAtomicNum() == 1:
            if atom.GetBonds()[0].GetOtherAtom(atom).GetAtomicNum() in polar_atoms:
                atom.SetProp("SASAClassName", "Polar") # mark as polar
        radii.append(symbol_radius[atom.GetSymbol().upper()])

    return radii


def ComputeSASA(coords:list, Rvdw:list):
    assert len(coords) == 3*len(Rvdw)
    sa = freesasa.calcCoord(coords, Rvdw)
    
    return sa.totalArea()


"""
def compute_sasa(mol):
    mol = Chem.AddHs(mol)

    # Get Van der Waals radii (angstrom)
    ptable = Chem.GetPeriodicTable()
    radii = [ptable.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()]

    # Compute solvent accessible surface area
    sa = rdFreeSASA.CalcSASA(mol, radii, confIdx=-1)
    
    return sa
"""