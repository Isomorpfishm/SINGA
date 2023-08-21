import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import oddt
from oddt.toolkits.ob import Molecule
import rdkit
from rdkit import Chem
import math
from math import sin, cos
import yaml



BASE_ANGLES = np.array((0, 180, 120, 109.5, 90), dtype=float)

#######################################################################################
################################## Helper Function ####################################
#######################################################################################

"""
Spatial functions included in ODDT
Mainly used by other modules, but can be accessed directly.
"""

def angle(p1, p2, p3):
    """Returns an angle from a series of 3 points (point #2 is centroid).
    Angle is returned in degrees.

    Parameters
    ----------
    p1, p2, p3 : numpy arrays, shape = [n_points, n_dimensions]
        Triplets of points in n-dimensional space, aligned in rows.

    Returns
    -------
    angles : numpy array, shape = [n_points]
        Series of angles in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2
    return angle_2v(v1, v2)


def angle_2v(v1, v2):
    """Returns an angle between two vecors.Angle is returned in degrees.

    Parameters
    ----------
    v1,v2 : numpy arrays, shape = [n_vectors, n_dimensions]
        Pairs of vectors in n-dimensional space, aligned in rows.

    Returns
    -------
    angles : numpy array, shape = [n_vectors]
        Series of angles in degrees
    """
    # better than np.dot(v1, v2), multiple vectors can be applied
    dot = (v1 * v2).sum(axis=-1)
    norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    return np.degrees(np.arccos(np.clip(dot/norm, -1, 1)))


def distance(x, y):
    """Computes distance between each pair of points from x and y.

    Parameters
    ----------
    x : numpy arrays, shape = [n_x, 3]
        Array of poinds in 3D

    y : numpy arrays, shape = [n_y, 3]
        Array of poinds in 3D

    Returns
    -------
    dist_matrix : numpy arrays, shape = [n_x, n_y]
        Distance matrix
    """
    return cdist(x, y)
    

def close_contacts(x, y, cutoff, x_column='coords', y_column='coords', cutoff_low=0.):
    """Returns pairs of atoms which are within close contac distance cutoff.
    The cutoff is semi-inclusive, i.e (cutoff_low, cutoff].

    Parameters
    ----------
    x, y : atom_dict-type numpy array
        Atom dictionaries generated by oddt.toolkit.Molecule objects.

    cutoff : float
        Cutoff distance for close contacts

    x_column, ycolumn : string, (default='coords')
        Column containing coordinates of atoms (or pseudo-atoms,
        i.e. ring centroids)

    cutoff_low : float (default=0.)
        Lower bound of contacts to find (exclusive). Zero by default.
        .. versionadded:: 0.6

    Returns
    -------
    x_, y_ : atom_dict-type numpy array
        Aligned pairs of atoms in close contact for further processing.
    """
    if len(x[x_column]) > 0 and len(x[x_column]) > 0:
        d = distance(x[x_column], y[y_column])
        index = np.argwhere((d > cutoff_low) & (d <= cutoff))
        return x[index[:, 0]], y[index[:, 1]]
    else:
        return x[[]], y[[]]


def _check_angles(angles, hybridizations, tolerance):
    """Helper function for checking if interactions are strict"""
    angles = np.nan_to_num(angles)  # NaN's throw warning on comparisons
    ideal_angles = np.take(BASE_ANGLES, hybridizations)[:, np.newaxis]
    lower_bound = ideal_angles - tolerance
    upper_bound = ideal_angles + tolerance
    return ((angles > lower_bound) & (angles < upper_bound)).any(axis=-1)


#######################################################################################
################################ Capture Interaction ##################################
#######################################################################################

def hbond_acceptor_donor(mol1:Molecule, 
                         mol2:Molecule, 
                         cutoff:float, 
                         tolerance:int=30, 
                         donor_exact:bool=False):
    """Returns pairs of acceptor-donor atoms, which meet H-bond criteria

    Parameters
    ----------
    mol1, mol2 : oddt.toolkit.Molecule object
        Molecules to compute H-bond acceptor and H-bond donor pairs

    cutoff : float, (default=3.5)
        Distance cutoff for A-D pairs

    tolerance : int, (default=30)
        Range (+/- tolerance) from perfect direction defined by acceptor/donor hybridization
        in which H-bonds are considered as strict.
    donor_exact : bool
        Use exact protonation states for donors, i.e. require Hs on donor.
        By default ODDT implies some tautomeric structures as protonated,
        even if there is no H on specific atom.

    Returns
    -------
    a, d : atom_dict-type numpy array
        Aligned arrays of atoms forming H-bond, firstly acceptors,
        secondly donors.

    angle1, angle2 : numpy array
        Aligned arrays of angles forming H-bond

    strict : numpy array, dtype=bool
        Boolean array align with atom pairs, informing whether atoms
        form 'strict' H-bond (pass all angular cutoffs). If false,
        only distance cutoff is met, therefore the bond is 'crude'.
    """
    donor_mask = mol2.atom_dict['isdonor']
    if donor_exact:
        donor_mask = donor_mask & (mol2.atom_dict['numhs'] > 0)
    a, d = close_contacts(mol1.atom_dict[mol1.atom_dict['isacceptor']],
                          mol2.atom_dict[donor_mask],
                          cutoff)
    # skip empty values
    if len(a) > 0 and len(d) > 0:
        angle1 = angle(d['coords'][:, np.newaxis, :],
                       a['coords'][:, np.newaxis, :],
                       a['neighbors'])
        angle2 = angle(a['coords'][:, np.newaxis, :],
                       d['coords'][:, np.newaxis, :],
                       d['neighbors'])
        strict = (_check_angles(angle1, a['hybridization'], tolerance) &
                  _check_angles(angle2, d['hybridization'], tolerance))
        return a, d, angle1, angle2, strict
    else:
        return a, d, np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=bool)


def halogenbond_acceptor_halogen(mol1:Molecule, 
                                 mol2:Molecule, 
                                 cutoff:float, 
                                 tolerance:int=30):
    """Returns pairs of acceptor-halogen atoms, which meet halogen bond criteria

    Parameters
    ----------
    mol1, mol2 : oddt.toolkit.Molecule object
        Molecules to compute halogen bond acceptor and halogen pairs

    cutoff : float, (default=4)
        Distance cutoff for A-H pairs

    tolerance : int, (default=30)
        Range (+/- tolerance) from perfect direction defined by atoms hybridization
        in which halogen bonds are considered as strict.

    Returns
    -------
    a, h : atom_dict-type numpy array
        Aligned arrays of atoms forming halogen bond, firstly acceptors,
        secondly halogens

    angle1, angle2 : numpy array
        Aligned arrays of angles forming X-bond
        
    strict : numpy array, dtype=bool
        Boolean array align with atom pairs, informing whether atoms
        form 'strict' halogen bond (pass all angular cutoffs). If false,
        only distance cutoff is met, therefore the bond is 'crude'.
    """
    a, h = close_contacts(mol1.atom_dict[mol1.atom_dict['isacceptor']],
                          mol2.atom_dict[mol2.atom_dict['ishalogen']],
                          cutoff)
    # skip empty values
    if len(a) > 0 and len(h) > 0:
        angle1 = angle(h['coords'][:, np.newaxis, :],
                       a['coords'][:, np.newaxis, :],
                       a['neighbors'])
        angle2 = angle(a['coords'][:, np.newaxis, :],
                       h['coords'][:, np.newaxis, :],
                       h['neighbors'])
        strict = (_check_angles(angle1, a['hybridization'], tolerance) &
                  _check_angles(angle2, np.ones_like(h['hybridization']), tolerance))
        return a, h, angle1, angle2, strict
    else:
        return a, h, np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=bool)


def salt_bridge_plus_minus(mol1:Molecule, 
                           mol2:Molecule, 
                           cutoff:float, 
                           cation_exact:bool=False, 
                           anion_exact:bool=False):
    """Returns pairs of plus-mins atoms, which meet salt bridge criteria

    Parameters
    ----------
    mol1, mol2 : oddt.toolkit.Molecule object
        Molecules to compute plus and minus pairs

    cutoff : float, (default=4)
        Distance cutoff for A-H pairs

    cation_exact, anion_exact : bool
        Requires interacting atoms to have non-zero formal charge.

    Returns
    -------
    plus, minus : atom_dict-type numpy array
        Aligned arrays of atoms forming salt bridge, firstly plus, secondly minus

    """
    cation_map = mol1.atom_dict['isplus']
    if cation_exact:
        cation_map = cation_map & (mol1.atom_dict['formalcharge'] > 0)
    anion_map = mol2.atom_dict['isminus']
    if anion_exact:
        anion_map = anion_map & (mol2.atom_dict['formalcharge'] < 0)
    m1_plus, m2_minus = close_contacts(mol1.atom_dict[cation_map],
                                       mol2.atom_dict[anion_map],
                                       cutoff)
    return m1_plus, m2_minus



#######################################################################################
##################################### Extraction ######################################
#######################################################################################

def hbond_oddt(protein:Molecule, 
               ligand:Molecule, 
               cutoff:float, 
               tolerance:int=30, 
               mol1_exact:bool=False, 
               mol2_exact:bool=False):
    """
    a1, d2 : protein as acceptor, ligand as donor
    
    angle1 = angle(a1d2[1]['coords'][:, np.newaxis, :],
                   a1d2[0]['coords'][:, np.newaxis, :],
                   a1d2[0]['neighbors'])
    angle2 = angle(a1d2[0]['coords'][:, np.newaxis, :],
                   a1d2[1]['coords'][:, np.newaxis, :],
                   a1d2[1]['neighbors'])
                   
    Interest: angle*[i][0]
    """
    a1, d2, _, angle1, _ = hbond_acceptor_donor(protein, ligand, cutoff, tolerance, mol2_exact)
    
    """
    a2, d1 : protein as donor, ligand as acceptor
    
    angle1 = angle(a2d1[1]['coords'][:, np.newaxis, :],
                   a2d1[0]['coords'][:, np.newaxis, :],
                   a2d1[0]['neighbors'])
    angle2 = angle(a2d1[0]['coords'][:, np.newaxis, :],
                   a2d1[1]['coords'][:, np.newaxis, :],
                   a2d1[1]['neighbors'])
    """
    a2, d1, _, angle2, _ = hbond_acceptor_donor(ligand, protein, cutoff, tolerance, mol1_exact)
    
    try:
        return np.concatenate((a1, d1)), np.concatenate((d2, a2)), np.concatenate((angle1, angle2))
    except ValueError:
        if angle1.shape[0] == 0 and angle2.shape[0] != 0:
            return np.concatenate((a1, d1)), np.concatenate((d2, a2)), angle2
        elif angle1.shape[0] != 0 and angle2.shape[0] != 0:
            return np.concatenate((a1, d1)), np.concatenate((d2, a2)), angle1
        else:
            return np.concatenate((a1, d1)), np.concatenate((d2, a2)), np.array([], dtype=float)


def xbond_oddt(protein:Molecule, 
               ligand:Molecule, 
               cutoff:float, 
               tolerance:int=30, 
               mol1_exact:bool=False, 
               mol2_exact:bool=False):
    """
    a1, h2 : protein as acceptor, ligand as halogen donor

    angle1 = angle(a1h2[1]['coords'][:, np.newaxis, :],
                   a1h2[0]['coords'][:, np.newaxis, :],
                   a1h2[0]['neighbors'])
    angle2 = angle(a1h2[0]['coords'][:, np.newaxis, :],
                   a1h2[1]['coords'][:, np.newaxis, :],
                   a1h2[1]['neighbors'])
    """
    a1, h2, _, angle1, _ = halogenbond_acceptor_halogen(protein, ligand, cutoff, tolerance)

    """    
    a2h1 : protein as donor, ligand as halogen acceptor

    angle1 = angle(a2h1[1]['coords'][:, np.newaxis, :],
                   a2h1[0]['coords'][:, np.newaxis, :],
                   a2h1[0]['neighbors'])
    angle2 = angle(a2h1[0]['coords'][:, np.newaxis, :],
                   a2h1[1]['coords'][:, np.newaxis, :],
                   a2h1[1]['neighbors'])
    """
    a2, h1, _, angle2, _ = halogenbond_acceptor_halogen(ligand, protein, cutoff, tolerance)
    
    try:
        return np.concatenate((a1, h1)), np.concatenate((h2, a2)), np.concatenate((angle1, angle2))
    except ValueError:
        if angle1.shape[0] == 0 and angle2.shape[0] != 0:
            return np.concatenate((a1, h1)), np.concatenate((h2, a2)), angle2
        elif angle1.shape[0] != 0 and angle2.shape[0] != 0:
            return np.concatenate((a1, h1)), np.concatenate((h2, a2)), angle1
        else:
            return np.concatenate((a1, h1)), np.concatenate((h2, a2)), np.array([], dtype=float)


def hphob_oddt(protein:Molecule, ligand:Molecule, cutoff:float):
    h1, h2 = close_contacts(protein.atom_dict[protein.atom_dict['ishydrophobe']], 
                            ligand.atom_dict[ligand.atom_dict['ishydrophobe']], 
                            cutoff)
    return h1, h2
    
    
def sbridge_oddt(protein:Molecule, 
                 ligand:Molecule, 
                 cutoff:float, 
                 mol1_exact:bool=False, 
                 mol2_exact:bool=False):
    """
    p1, l2: protein positive, ligand negative
    """
    protein_plus, ligand_minus = salt_bridge_plus_minus(protein, ligand, cutoff)
    
    """
    l1, p2: ligand positive, protein negative
    """
    ligand_plus, protein_minus = salt_bridge_plus_minus(ligand, protein, cutoff)

    return np.concatenate((protein_plus, protein_minus)), np.concatenate((ligand_minus, ligand_plus))
    

def pistack_oddt(protein:Molecule, 
                 ligand:Molecule, 
                 cutoff:float, 
                 tolerance:int=30):
    
    r1, r2 = close_contacts(protein.ring_dict, ligand.ring_dict, cutoff, x_column='centroid', y_column='centroid')
    
    if len(r1) > 0 and len(r2) > 0:
        angle1 = angle_2v(r1['vector'], r2['vector'])
        angle2 = angle(r1['vector'] + r1['centroid'],
                       r1['centroid'],
                       r2['centroid'])
        angle3 = angle(r2['vector'] + r2['centroid'],
                       r2['centroid'],
                       r1['centroid'])
        return r1, r2, angle1, angle2
    else:
        return r1, r2, np.array([], dtype=bool), np.array([], dtype=bool)


def pication_oddt(protein:Molecule, 
                  ligand:Molecule, 
                  cutoff:float, 
                  tolerance:int=30, 
                  cation_exact:bool=False):

    cation_map = ligand.atom_dict['isplus']
    
    if cation_exact:
        cation_map = cation_map & (ligand.atom_dict['formalcharge'] > 0)
    
    r1, plus2 = close_contacts(protein.ring_dict,
                               ligand.atom_dict[cation_map],
                               cutoff,
                               x_column='centroid')
    
    if len(r1) > 0 and len(plus2) > 0:
        angle1 = angle_2v(r1['vector'], plus2['coords'] - r1['centroid'])
        return r1, plus2, angle1
    else:
        return r1, plus2, np.array([], dtype=bool)
