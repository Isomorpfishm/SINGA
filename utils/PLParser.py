###################################################################################
# Disclaimer:                                                                     #
# This Python script is extracted from Pocket2Mol project, which is accessible at # 
# https://github.com/pengxingang/Pocket2Mol/blob/main/utils/protein_ligand.py     #
###################################################################################

import os
from typing import List
import numpy as np
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdmolops import AddHs
from rdkit.Chem import ChemicalFeatures
import oddt
import biopandas.pdb as bpdb

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BondType.names.keys())}


class PDBProtein(object):
    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto'):
        super().__init__()
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []
        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break   # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass
        
        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name   # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=np.longlong),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=np.bool),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.longlong)
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.longlong),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        
        return selected

    def query_residues_ligand(self, ligand, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        for center in ligand['pos']:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        
        return selected

    def residues_to_pdb_block(self, residues, name='POCKET'):
        block =  "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        
        return block


def parse_pdbbind_index_file(path):
    pdb_id = []
    with open(path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if line.startswith('#'): continue
        pdb_id.append(line.split()[0])
    
    return pdb_id


def parse_sdf_file(path):
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=False)))
    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.longlong)
    
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    with open(path, 'r') as f:
        sdf = f.read()

    sdf = sdf.splitlines()
    num_atoms, num_bonds = map(int, [sdf[3][0:3], sdf[3][3:6]])
    assert num_atoms == rd_num_atoms

    ptable = Chem.GetPeriodicTable()
    element, pos = [], []
    accum_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    accum_mass = 0.0
    
    for atom_line in map(lambda x:x.split(), sdf[4:4+num_atoms]):
        x, y, z = map(float, atom_line[:3])
        symb = atom_line[3]
        atomic_number = ptable.GetAtomicNumber(symb.capitalize())
        element.append(atomic_number)
        pos.append([x, y, z])
        
        atomic_weight = ptable.GetAtomicWeight(atomic_number)
        accum_pos += np.array([x, y, z]) * atomic_weight
        accum_mass += atomic_weight

    center_of_mass = np.array(accum_pos / accum_mass, dtype=np.float32)

    element = np.array(element, dtype=np.int64)
    pos = np.array(pos, dtype=np.float32)

    BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
    bond_type_map = {
        1: BOND_TYPES[BondType.SINGLE],
        2: BOND_TYPES[BondType.DOUBLE],
        3: BOND_TYPES[BondType.TRIPLE],
        4: BOND_TYPES[BondType.AROMATIC],
    }
    
    row, col, edge_type = [], [], []
    for bond_line in sdf[4+num_atoms:4+num_atoms+num_bonds]:
        start, end = int(bond_line[0:3])-1, int(bond_line[3:6])-1
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond_type_map[int(bond_line[6:9])]]

    edge_index = np.array([row, col], dtype=np.longlong)
    edge_type = np.array(edge_type, dtype=np.longlong)

    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
    }
    
    return data


class StructureDual(object):
    """A helper class that prepares dual format (RDKit rdChem.Mol and ODDT Ob.Molecule) of a given structure"""
    
    def __init__(self, path:str, isProtein:bool=False, addHs:bool=True, removeHs:bool=False, sanitise:bool=False):
        super().__init__()
        if sanitise is False and removeHs is True:
            raise ArgumentError("Not allowed to remove Hs while sanitisation is false!")
        if path is None:
            raise ArgumentError("Path to structure not given!")
        if addHs == removeHs:
            raise ArgumentError("Adding and removing hydrogens conflict")

        self.addHs = addHs
        self.removeHs = removeHs
        self.sanitise = sanitiseRet
        self.path = path
        self.isProtein = isProtein
        self.ppdb = bpdb.PandasPdb()
            
        if not isProtein:
            self.fileType = str(path.split(".")[-1])
        else:
            self.fileType = 'pdb'
            
    def parse_to_rdkit(self):
        if self.isProtein:
            protein = Chem.rdmolfiles.MolFromPDBFile(self.path, self.removeHs, self.sanitise)
            if self.addHs:
                protein = AddHs(protein, addCoords=True)
            return protein
        else:
            if self.fileType == 'sdf':
                ligand = Chem.SDMolSupplier(self.path)[0]
            elif self.fileType == 'mol2':
                ligand = Chem.rdmolfiles.MolFromMol2File(self.path, self.removeHs, self.sanitise)
                
            if self.addHs:
                ligand = AddHs(ligand, addCoords=True)
            return ligand
        
    def parse_to_oddt(self) -> Molecule:
        if self.isProtein:
            protein = next(oddt.toolkit.readfile('pdb', self.path))
            protein.protein = True
            if self.removeHs:
                protein.removeh()
            return protein
        else:
            ligand = next(oddt.toolkit.readfile(self.fileType, self.path))
            if self.removeHs:
                ligand.removeh()
            return ligand 
            
    def RetrieveCoords(self, consideredIdx=None) -> List:
        molecule = self.parse_to_rdkit()
        pos = []
        
        if consideredIdx is None:
            consideredIdx = [i for i in range(molecule.GetNumAtoms())]
        
        for idx, atom in enumerate(molecule.GetAtoms()):
            if idx not in consideredIdx:
                continue
            else:
                coords = molecule.GetConformer().GetAtomPosition(idx)
                pos.append([coords.x, coords.y, coords.z])
        return pos
        
    def RetreiveAtomNames(self) -> List:
        self.ppdb.read_pdb(self.path)
        atom_df = self.ppdb.df['ATOM'][self.ppdb.df['ATOM']['element_symbol'] != 'H']
        list_atom_name = atom_df['atom_name'].tolist()
        return list_atom_name
        