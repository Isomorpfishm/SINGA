import biopandas.pdb as bpdb
import oddt
from PLInteraction import *

protein_path = '../example/7cff_protein.pdb'
ligand_path = '../example/7cff_ligand.sdf'

ppdb = bpdb.PandasPdb()
ppdb.read_pdb(protein_path)

atom_df = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H']
list_atom_name = atom_df['atom_name'].tolist()

protein = next(oddt.toolkits.ob.readfile('pdb', protein_path))
protein.removeh()
protein.protein = True

ligand = next(oddt.toolkits.ob.readfile('sdf', ligand_path))
ligand.removeh()


#get_bonds_protein_ligand(protein, ligand, 4.0, list_atom_name)
alpha, beta, gamma, delta = get_bonds_protein_ligand(protein, ligand, 4.0, list_atom_name)

