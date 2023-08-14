import math
from math import sin, cos
from scipy.spatial.distance import cdist
import numpy as np


"""
Spatial functions included in ODDT
Mainly used by other modules, but can be accessed directly.
"""

def angle(p1, p2, p3):
    """Returns an angle from a series of 3 points (point #2 is centroid).
    Angle is returned in degrees.

    Parameters
    ----------
    p1,p2,p3 : numpy arrays, shape = [n_points, n_dimensions]
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
    

def close_contacts(x, y, cutoff, x_column='coords', y_column='coords',
                   cutoff_low=0.):
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


def hbond_acceptor_donor(mol1, mol2, cutoff=3.5, tolerance=30, donor_exact=False):
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
        return a, d, np.array([], dtype=bool), np.array([], dtype=bool), np.array([], dtype=bool)

def halogenbond_acceptor_halogen(mol1,
                                 mol2,
                                 tolerance=30,
                                 cutoff=4):
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
        return a, h, np.array([], dtype=bool), np.array([], dtype=bool), np.array([], dtype=bool)


def salt_bridge_plus_minus(mol1, mol2, cutoff=4, cation_exact=False, anion_exact=False):
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


def calculate_atomic_rmsd(reference_mol, target_mol):
    reference_atoms = reference_mol.GetAtoms()
    target_atoms = target_mol.GetAtoms()

    if len(reference_atoms) != len(target_atoms):
        raise ValueError("The number of atoms in the molecules does not match.")

    squared_sum = 0.0
    count = 0

    for ref_atom, tgt_atom in zip(reference_atoms, target_atoms):
        ref_coords = ref_atom.GetPosition()
        tgt_coords = tgt_atom.GetPosition()

        squared_sum += (ref_coords.x - tgt_coords.x) ** 2
        squared_sum += (ref_coords.y - tgt_coords.y) ** 2
        squared_sum += (ref_coords.z - tgt_coords.z) ** 2
        count += 3

    rmsd = (squared_sum / count) ** 0.5

    return rmsd


BASE_ANGLES = np.array((0, 180, 120, 109.5, 90), dtype=float)

settings = {}
settings['nonpolar'] = {6:1.7, 9:1.47, 17:1.75, 35:1.85, 53:1.98}
settings['hbond_dist_cut'] = 3.5
settings['hbond_angle_cut'] = 180.0
settings['hbond_dist_bound'] = 2.5
settings['hbond_angle_bound'] = 90.0

settings['hphob_dist_cut'] = 4.0
settings['hphob_dist_bound'] = 1.25

settings['contact_scale_cut'] = 1.75
settings['contact_scale_opt'] = 1.25

settings['sbridge_dist_cut'] = 5.0
settings['sbridge_dist_bound'] = 3.25
settings['saltbridge_resonance'] = True

settings['pistack_dist_cut'] = 6.5
settings['pistack_dist_bound'] = 3.8
settings['pistack_angle_bound'] = 60.0
settings['pistack_angle_cut'] = 90.0

settings['pication_dist_cut'] = 6.5
settings['pication_dist_bound'] = 4.3
settings['pication_angle_cut'] = 90.0
settings['pication_angle_bound'] = 30.0

settings['xbond_dist_cut'] = 5.0
settings['xbond_angle_cut'] = 150.0
settings['xbond_dist_bound'] = 3.5
settings['xbond_angle_bound'] = 135.0

settings['acceptor_metal_dist_cut'] = 4.0
settings['pimetal_dist_cut'] = 5.0

