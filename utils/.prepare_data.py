# Main functionality
# ==================
# 1. Find RMSD of docked ligand poses from crystal structure
# 2. Retrieve molecular features of protein and ligand
# 3. Prepare node and edge features (ODDT)
# 4. Dump all relevant info to pickle file 
#
# Note: Compatible with LeDock generated (and corrected) poses only

import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolAlign
import prolif as plf

from spyrmsd import io, rmsd
from spyrmsd.molecule import Molecule
from spyrmsd.rmsd import rmsdwrapper

from tqdm import tqdm
import os, sys, glob
import pickle, yaml
import warnings
import argparse
import traceback
import logging

from Mol2Graph import *
import PLInteraction




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path', type=str,
                        default='../download/v2020-other-PL')
    parser.add_argument('--dok_path', type=str,
                        default='./dataset/general')
    parser.add_argument('--mesh_path', type=str,
                        default='./dataset/general_mesh')
    parser.add_argument('--config', type=str,
                        default='./config/config.yaml')
    parser.add_argument('--outdir', type=str, 
                        default='./features')
    args = parser.parse_args()

    if args.ref_path.split("/")[-1] not in ['v2020-other-PL', 'refined-set']:
        raise FileNotFoundError("Sorry, option for `--ref_path` not supported / does not exist.")
    if args.dok_path.split("/")[-1] not in ['general', 'refined']:
        raise FileNotFoundError("Sorry, option for `--dok_path` not supported / does not exist.")
    if args.mesh_path.split("/")[-1] not in ['general_mesh', 'refined_mesh']:
        raise FileNotFoundError("Sorry, option for `--mesh_path` not supported / does not exist.")
        
    print("\n")
    outDirisExist = os.path.exists(args.outdir)
    if not outDirisExist:
       # Create a new directory because it does not exist
       os.makedirs(args.outdir)
       print("The outdir directory is created!")

    outDirisExist = os.path.isfile(args.config)
    if not outDirisExist:
       warnings.warn("Configuration YAML file does not exist or is not specified. Assumed all complexes are utilised.")
       meshNotExist = list()
    else:
        with open(args.config, 'rb') as file:
            ymlDict = yaml.safe_load(file)
            meshNotExist = ymlDict['pymeshSkippedComplex']
            print("Reading configuration YAML file...")
            print(f"Skipped {len(meshNotExist)} complexes for PyMesh file does not exist.")
    
    
    """
    Part 2: Fix LeDock output docked files
            maybe carry out using the bash script modifyDOK.sh (?)
    
    """
    
    
    
    
    """
    Part 3: Find docked poses' RMSD from crystal structure using spyrmsd.
            This RMSD value will be out value `y`.
    """
    allAvailableComplex = os.listdir(args.ref_path)
    print(f"Found {len(allAvailableComplex)-2} in {args.ref_path.split('/')[-1]} for RMSD calculation.")
    
    #allAvailableComplex = ['9icd', '4kqo', '5qj2']
    allAvailableComplex = ['9icd']
    skippedComplex, rmsdDict = [], {}
    print("\nProcessing complexes RMSD...")
    
    for i in tqdm(range(len(allAvailableComplex))):
        if (str(allAvailableComplex[i]) in ['index', 'readme']) or (str(allAvailableComplex[i]) in meshNotExist): continue

        refMol = Chem.SDMolSupplier(args.ref_path + "/" + \
                                                 str(allAvailableComplex[i]) + "/" + \
                                                 str(allAvailableComplex[i]) + "_ligand.sdf")
        refMol = Molecule.from_rdkit(refMol[0])
        refMol.strip() # remove hydrogen atoms

        #allDokPoses = glob.glob(args.dok_path + "/" + str(allAvailableComplex[i]) + "*_dock*.pdb")
        allDokPoses = glob.glob(args.ref_path + "/" + \
                                str(allAvailableComplex[i]) + "/" + \
                                str(allAvailableComplex[i]) + "*_dock*.pdb")
        
        if len(allDokPoses) == 0: skippedComplex.append(allAvailableComplex[i])
        else:
            for j in range(len(allDokPoses)):
                dokMol = Chem.rdmolfiles.MolFromPDBFile(allDokPoses[j], removeHs=True)
                dokMol = Molecule.from_rdkit(dokMol)
                dokMol.strip() # remove hydrogen atoms
                
                RMSD = rmsdwrapper(refMol, dokMol)[0]
                print(f'Complex {allAvailableComplex[i]} docked poses {j+1}: RMSD = {round(RMSD, 3)}')
                
                rmsdDict[str(allAvailableComplex[i]) + "_" + str(j+1)] = round(RMSD, 3)
            

    """
    Part 4: Retrieve molecular features
    """
    
    ligRMSDList, repDict, ligDict = [], {}, {}
    print("\nRetrieving molecular features...")
     
    for i in tqdm(range(len(allAvailableComplex))):
        if (allAvailableComplex[i] in ['readme', 'index'] + skippedComplex) or (str(allAvailableComplex[i]) in meshNotExist): continue
        repFile = args.ref_path + "/" + \
                  str(allAvailableComplex[i]) + "/" + \
                  str(allAvailableComplex[i]) + "_pocket.pdb"
        repDict[str(allAvailableComplex[i])] = RetrieveMolecularFeatures([repFile], [0.0], removeHs=False)[0]
        
        #allDokPoses = glob.glob(args.dok_path + "/" + str(allAvailableComplex[i]) + "*_dock*.pdb")
        allDokPoses = glob.glob(args.ref_path + "/" + \
                                str(allAvailableComplex[i]) + "/" + \
                                str(allAvailableComplex[i]) + "*_dock*.pdb")
        
        for j in range(len(allDokPoses)):
            ligFile = allDokPoses[j]
            ligRMSD = rmsdDict[str(allAvailableComplex[i]) + "_" + str(j+1)]
            ligDict[str(allAvailableComplex[i]) + "_" + str(j+1)] = RetrieveMolecularFeatures([ligFile], [ligRMSD], removeHs=False)[0]
    
    
    
    """
    Part 5(A): Prepare node and edge features:
               Protein-ligand interactions as recognised by PLIP.
               (?) oddt or plip or prolif though ...
               
    iaDict = {}
    totalIA = ["Hydrophobic", "HBAcceptor", "HBDonor"   , "XBAcceptor"   , "XBDonor",
               "Cationic"   , "Anionic"   , "CationPi"  , "PiCation"     , "FaceToFace", 
               "EdgeToFace" , "PiStacking", "MetalDonor", "MetalAcceptor", "VdWContact"]    
    print("\nPreparing node and edge features")
    
    for i in tqdm(range(len(allAvailableComplex))):
        if (allAvailableComplex[i] in ['readme', 'index'] + skippedComplex) or (str(allAvailableComplex[i]) in meshNotExist): continue
        else:
            mol = Chem.MolFromPDBFile(args.ref_path + "/" + \
                                      str(allAvailableComplex[i]) + "/" + \
                                      str(allAvailableComplex[i]) + "_pocket.pdb", removeHs=False)
            prot = plf.Molecule(mol)
            
            allDokPoses = glob.glob(args.dok_path + "/" + str(allAvailableComplex[i]) + "*_dock*.pdb")
            allDokPoses = glob.glob(args.ref_path + "/" + \
                                    str(allAvailableComplex[i]) + "/" + \
                                    str(allAvailableComplex[i]) + "*_dock*.pdb")
        
        for j in range(len(allDokPoses)):
            mol = Chem.MolFromPDBFile(allDokPoses[j], removeHs=False)
            lig = plf.Molecule(mol)
    
            fp = plf.Fingerprint()
            fp.run_from_iterable([lig], prot)
            df = fp.to_dataframe()
            occ = df.mean()
            occ = occ.reset_index()

            ### map each color to an integer
            mapping = {}
            for x in range(len(totalIA)):
              mapping[totalIA[x]] = x

            one_hot_encode = []

            for someIA in np.array(occ['interaction']).tolist():
                arr = list(np.zeros(len(totalIA), dtype = int))
                arr[mapping[someIA]] = 1
                one_hot_encode.append(arr)

            occ['OneHotInteraction'] = one_hot_encode
            occ[['ResidueID', 'ResidueNum', 'dot', 'ChainID']] = occ['protein'].str.extract(r'(\D{3})(\d+)(\.)(\D)')
            occ.drop(['dot', 'ligand'], axis=1, inplace=True)
    
            iaDict[str(allAvailableComplex[i]) + "_" + str(j+1)] = occ
    """
    
    
    
    """
    Part 5(B): Prepare edge features for PL Interactions
               using ODDT package. Export data to iaDict
    """ 
    print("\nRetrieving PL interaction (edge) features...")
    
    iaDict = {}
    for i in tqdm(range(len(allAvailableComplex))):
        if (allAvailableComplex[i] in ['readme', 'index'] + skippedComplex) or (str(allAvailableComplex[i]) in meshNotExist): continue
        repFile = args.ref_path + "/" + \
                  str(allAvailableComplex[i]) + "/" + \
                  str(allAvailableComplex[i]) + "_protein.pdb"
        allDokPoses = glob.glob(args.ref_path + "/" + \
                                str(allAvailableComplex[i]) + "/" + \
                                str(allAvailableComplex[i]) + "*_dock*.pdb")
        for j in range(len(allDokPoses)):
            ligFile = allDokPoses[j]
            protein, ligand, max_len, lig_len = PLInteraction.read_prolig(repFile, ligFile)
            fp, fl = PLInteraction.fingerprint(protein, ligand, max_len, lig_len)
            iaDict[str(allAvailableComplex[i]) + "_" + str(j+1)] = [fp, fl]
    
    
    """
    Part 6: Processing PyMesh data for proteins
    """
    print("\nRetrieving protein mesh files...")
    
    meshDict = {}
    for i in tqdm(range(len(allAvailableComplex))):
        if (allAvailableComplex[i] in ['readme', 'index'] + skippedComplex) or (str(allAvailableComplex[i]) in meshNotExist): continue
        with open(args.mesh_path + "/" + str(allAvailableComplex[i]) + "_protein.ply", 'r') as file:
            meshDict[str(allAvailableComplex[i])] = yaml.safe_load(file)
    
    
    """
    Part 7: Conclusion
    """
    print("\nSaving pickle and shutting down processes...\n")
    
    allDict = {}
    allDict['rmsdDict'] = rmsdDict
    allDict['repDict'] = repDict
    allDict['ligDict'] = ligDict
    allDict['iaDict'] = iaDict
    allDict['meshDict'] = meshDict
    ymlDict['processSkippedComplex'] = skippedComplex
    
    # Pickle dump
    file = open(args.outdir + '/PLComplex.pkl', 'wb')
    pickle.dump(allDict, file)
    file.close()
    
    with open(args.config, 'r') as file:
        _ymlDict = yaml.safe_load(file)
        _ymlDict.update(ymlDict)
        
    with open(args.config, 'w') as file:
        yaml.safe_dump(_ymlDict, file)
    
    print(f"Skipped complexes: {skippedComplex}")
    print("Process terminated successfully.\n")
    
    