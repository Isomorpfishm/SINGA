import numpy as np
from oddt.toolkits.ob import Molecule
from oddt.spatial import distance
from typing import List, Tuple

try:
    import PLIExtension as interactions
except:
    from utils import PLIExtension as interactions


def extract_atom_id_from_oddt_interractions(mol1_atoms_array, mol2_atoms_array) -> dict:
    dico = {}
    for mol1_atom, mol2_atom in zip(mol1_atoms_array, mol2_atoms_array):
        mol1_atom_id = int(mol1_atom[0])
        mol2_atom_id = int(mol2_atom[0])
        if mol1_atom_id not in dico.keys():
            dico[mol1_atom_id] = []
        dico[mol1_atom_id] += [mol2_atom_id]
    return dico


def extract_residu_id_from_oddt_interractions(protein_residus_array):
    list_residu = []
    if protein_residus_array.shape[0] != 0:
        for np_row in np.nditer(protein_residus_array):
            list_residu.append(np_row.tolist()[2])
    return list_residu


def atom_pair_in_dico(dico:dict, mol1_atom_id:str, mol2_atom_id:str):
    if mol1_atom_id in dico.keys():
        return mol2_atom_id in dico[mol1_atom_id]
    else:
        return False


def is_pi(res_name:str, atom_name:str) -> bool:
    if res_name == 'HIS':
        if (atom_name == 'CG' or atom_name == 'CD2' or atom_name == 'NE2'
                or atom_name == 'CE1' or atom_name == 'ND1'):
            return True
        else:
            return False
    elif res_name == 'PHE':
        if (atom_name == 'CG' or atom_name == 'CD2' or atom_name == 'CE2'
                or atom_name == 'CZ' or atom_name == 'CE1'
                or atom_name == 'CD1'):
            return True
        else:
            return False
    elif res_name == 'TYR':
        if (atom_name == 'CG' or atom_name == 'CD1' or atom_name == 'CE1'
                or atom_name == 'CE2' or atom_name == 'CD2'
                or atom_name == 'CZ'):
            return True
        else:
            return False
    elif res_name == 'TRP':
        if (atom_name == 'CG' or atom_name == 'CD1' or atom_name == 'NE1'
                or atom_name == 'CE2' or atom_name == 'CD2'
                or atom_name == 'CE3' or atom_name == 'CZ2'
                or atom_name == 'CZ3' or atom_name == 'CH2'):
            return True
        else:
            return False
    else:
        return False


def close_contact_to_dict(protein_close_contacts: np.ndarray,
                          ligand_close_contacts: np.ndarray) -> dict:
    dict_close_contacts = {}
    for protein_atom, ligand_atom in zip(protein_close_contacts, ligand_close_contacts):
        protein_atom_id = int(protein_atom[0])
        ligand_atom_id = int(ligand_atom[0])
        if ligand_atom_id not in dict_close_contacts.keys():
            dict_close_contacts[ligand_atom_id] = []
        dict_close_contacts[ligand_atom_id] += [[protein_atom_id, protein_atom, ligand_atom]]
    
    return dict_close_contacts


def remove_dupl_angles(protein_array:np.ndarray,
                       ligand_array:np.ndarray,
                       angles_array:np.ndarray,
                       intr_type:str = None):
    """
    It was found that ODDT contains bugs where duplicated pairs of interaction are observed.
    Hence, we have to remove duplicated angles when this situation is observed
    
    Args:
        protein_array (np.ndarray): Array of protein atoms found in interaction
        ligand_array (np.ndarray):
        angles_array (np.ndarray):
        intr_type (str): Different comparison approach for aromatic interaction (pi-stack and pi-cation)
        
    Output:
        angle_array (np.ndarray): Clean array of angles
    """
    to_be_removed = []
    assert protein_array.shape[0] == ligand_array.shape[0]

    for i in range(protein_array.shape[0]-1):
        if intr_type is None:
            if (protein_array[i][0], ligand_array[i][0]) == (protein_array[i+1][0], ligand_array[i+1][0]):
                to_be_removed.append(i+1)
            else:
                continue
        elif intr_type == 'pistack':
            if np.array( np.concatenate((protein_array[i][0], ligand_array[i][0])) == np.concatenate((protein_array[i+1][0], ligand_array[i+1][0])) ).all():
                to_be_removed.append(i+1)
            else:
                continue 
        elif intr_type == 'pication':
            if np.array( np.concatenate((protein_array[i][0], ligand_array[i][1])) == np.concatenate((protein_array[i+1][0], ligand_array[i+1][1])) ).all():
                to_be_removed.append(i+1)
            else:
                continue
        elif intr_type == 'pication_rev':
            if np.array( np.concatenate((protein_array[i][1], ligand_array[i][0])) == np.concatenate((protein_array[i+1][1], ligand_array[i+1][0])) ).all():
                to_be_removed.append(i+1)
            else:
                continue
    angles_array = np.delete(angles_array, to_be_removed, 0)
    return angles_array


def get_bonds_protein_ligand(protein:Molecule, 
                             ligand:Molecule,
                             cutoff:float,
                             list_atom_name:List[str]):
    """
    Returns the bond between the protein and the ligand regarding the cutoff.
    All ligand must have at least one edge with an protein's atom.

    Args:
        protein (Molecule): protein
        ligand (Molecule):  ligand
        cutoff (float): The maximal distance between two atoms to connect them with an edge.
        list_atom_name (List[str]): List of PDB atom name, use for pi interactions

    Output:
        Tuple(List, List, List, List):Protein to Ligand Edge Index, 
                                      Ligand to Protein Edge Index, 
                                      Protein to Ligand Edge Attr, 
                                      Ligand to Protein Edge Attr
    """
    close_contact_protein, close_contact_ligand = interactions.close_contacts(protein.atom_dict, ligand.atom_dict, cutoff=cutoff)

    hbond_protein, hbond_ligand, hbond_angles = interactions.hbond_oddt(protein, ligand, cutoff=cutoff)
    dico_hbond = extract_atom_id_from_oddt_interractions(hbond_protein, hbond_ligand)
    if len(hbond_angles) > 1:
        hbond_angles = remove_dupl_angles(hbond_protein, hbond_ligand, hbond_angles)
    
    xbond_protein, xbond_ligand, xbond_angles = interactions.xbond_oddt(protein, ligand, cutoff=cutoff)
    dico_xbond = extract_atom_id_from_oddt_interractions(xbond_protein, xbond_ligand)
    if len(xbond_angles) > 1:
        xbond_angles = remove_dupl_angles(xbond_protein, xbond_ligand, xbond_angles)

    hphob_protein, hphob_ligand = interactions.hphob_oddt(protein, ligand, cutoff=cutoff)
    dico_hphob = extract_atom_id_from_oddt_interractions(hphob_protein, hphob_ligand)

    sbridge_protein, sbridge_ligand = interactions.sbridge_oddt(protein, ligand, cutoff=cutoff)
    dico_sbridge = extract_atom_id_from_oddt_interractions(sbridge_protein, sbridge_ligand)

    # pi_stacking
    pistack_protein_residue, pistack_ligand, pistack_angles, _ = interactions.pistack_oddt(protein, ligand, cutoff=cutoff)
    list_residus_pistack = extract_residu_id_from_oddt_interractions(pistack_protein_residue)
    if len(pistack_angles) > 1:
        pistack_angles = remove_dupl_angles(pistack_protein_residue, pistack_ligand, pistack_angles, intr_type='pistack')

    # pi_cation
    pication_protein_residue, pication_ligand, pication_angles = interactions.pication_oddt(protein, ligand, cutoff=cutoff)
    list_residus_pication = extract_residu_id_from_oddt_interractions(pication_protein_residue)
    if len(pication_angles) > 1:
        pication_angles = remove_dupl_angles(pication_protein_residue, pication_ligand, pication_angles, intr_type='pication')
        
    # pi_cation - reversed
    pication_ligand_rev, pication_protein_residue_rev, pication_angles_rev = interactions.pication_oddt(ligand, protein, cutoff=cutoff)
    list_residus_pication_rev = extract_residu_id_from_oddt_interractions(pication_protein_residue_rev)
    if len(pication_angles_rev) > 1:
        pication_angles_rev = remove_dupl_angles(pication_protein_residue_rev, pication_ligand_rev, pication_angles_rev, intr_type='pication_rev')

    # pication info housekeeping
    list_residus_pication = list_residus_pication + list_residus_pication_rev
    pication_angles = np.concatenate((pication_angles, pication_angles_rev))
    
    protein_atom_to_res_dict = {}
    for np_row in np.nditer(protein.atom_dict):
        protein_atom_to_res_dict[np_row.tolist()[0]] = np_row.tolist()[9]

    p_to_l_edge_index = [[], []]
    p_to_l_edge_attr = []
    l_to_p_edge_index = [[], []]
    l_to_p_edge_attr = []
    dict_close_contacts = close_contact_to_dict(close_contact_protein, close_contact_ligand)
    dists = distance(protein.atom_dict['coords'], ligand.atom_dict['coords'])
    
    i, j, k, l = 0, 0, 0, 0
    angle_hbond, angle_xbond, angle_pistack, angle_pication = 0.0, 0.0, 0.0, 0.0
    for ligand_atom_id in range(len(ligand.atoms)):
        if ligand_atom_id in dict_close_contacts.keys():
            for close_contact in dict_close_contacts[ligand_atom_id]:
                protein_atom_id = close_contact[1][0]
                protein_atom = close_contact[1]
                ligand_atom = close_contact[2]
                dist = distance([protein_atom[1]], [ligand_atom[1]])[0][0]
                protein_atom_res = protein_atom_to_res_dict[protein_atom_id]
                protein_atom_name = list_atom_name[protein_atom_id]
                res_name = protein_atom[11]

                atom_is_pi = is_pi(res_name, protein_atom_name)

                is_hbond = atom_pair_in_dico(dico_hbond, protein_atom_id, ligand_atom_id)
                is_xbond = atom_pair_in_dico(dico_xbond, protein_atom_id, ligand_atom_id)
                is_hphob = atom_pair_in_dico(dico_hphob, protein_atom_id, ligand_atom_id)
                is_sbridge = atom_pair_in_dico(dico_sbridge, protein_atom_id, ligand_atom_id)
                is_pistack = protein_atom_res in list_residus_pistack and atom_is_pi
                is_pication = protein_atom_res in list_residus_pication and atom_is_pi

                p_to_l_edge_index[0] += [int(protein_atom_id)]
                p_to_l_edge_index[1] += [int(ligand_atom_id)]
                l_to_p_edge_index[0] += [int(ligand_atom_id)]
                l_to_p_edge_index[1] += [int(protein_atom_id)]
                
                if is_hbond:
                    if i < len(hbond_angles):
                        angle_hbond = hbond_angles[i][0]
                        i += 1
                if is_xbond:
                    if j < len(xbond_angles):
                        angle_xbond = xbond_angles[j][0]
                        j += 1               
                if is_pistack:
                    if k < len(pistack_angles):
                        angle_pistack = pistack_angles[k]
                        k += 1
                if is_pication:
                    if l < len(pication_angles):
                        angle_pication = pication_angles[l]
                        l += 1
                                                        
                p_to_l_edge_attr += [[dist, angle_hbond, angle_xbond, angle_pistack, angle_pication, 
                                      is_hbond, is_xbond, is_hphob, is_sbridge,
                                      is_pistack, is_pication]]
                l_to_p_edge_attr += [[dist, angle_hbond, angle_xbond, angle_pistack, angle_pication, 
                                      is_hbond, is_xbond, is_hphob, is_sbridge,
                                      is_pistack, is_pication]]
        else:
            closer_protein_atom_id = np.argmin(dists[:, ligand_atom_id])
            dist = dists[closer_protein_atom_id, ligand_atom_id]
            p_to_l_edge_index[0] += [int(closer_protein_atom_id)]
            p_to_l_edge_index[1] += [int(ligand_atom_id)]
            l_to_p_edge_index[0] += [int(ligand_atom_id)]
            l_to_p_edge_index[1] += [int(closer_protein_atom_id)]
            p_to_l_edge_attr += [[dist, 0.0, 0.0, 0.0, 0.0, False, False, False, False, False, False]]
            l_to_p_edge_attr += [[dist, 0.0, 0.0, 0.0, 0.0, False, False, False, False, False, False]]
    
    return p_to_l_edge_index, l_to_p_edge_index, p_to_l_edge_attr, l_to_p_edge_attr
