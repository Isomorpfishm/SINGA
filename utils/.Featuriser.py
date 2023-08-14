#################################################################
# Main functionality                                            # 
# ==================                                            # 
# 1. Find RMSD of docked ligand poses from crystal structure    #
# 2. Retrieve molecular features of protein and ligand          #
# 3. Prepare node and edge features (ODDT)                      #
# 4. Dump all relevant info to pickle file                      #
#                                                               #
#################################################################

import rdkit
from rdkit import Chem
from rdkit.Chem.rdmolops import AddHs
import oddt
from oddt.docking.AutodockVina import autodock_vina
import torch

from tqdm import tqdm
import os, glob
import pickle, yaml
import random
import warnings
import argparse
import traceback
import logging
import itertools

from misc import *
import PLInteraction
from PLParser import StructureDual, parse_sdf_file
from PLFeature import ComputeSASA, ClassifyAtoms, RetrieveMolecularFeatures



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='./config/config.yml')
    parser.add_argument('--outdir', type=str, 
                        default='./dataset/crossdocked_vina10')
    args = parser.parse_args()
    
    # Logging
    log_dir = args.outdir
    logger = get_logger('feature_extract_log', log_dir)
    logger.info(args)
    
    print("\n")
    outDirExists = os.path.exists(args.outdir)
    if not outDirExists:
       os.makedirs(args.outdir)
       logger.info("Output directory {args.outdir} is created.")

    # Load config
    outDirExists = os.path.isfile(args.config)
    if not outDirExists:
        raise FileNotFoundError("Configuration YML (config.yml) file does not exist or is not specified.")
    else:
        logger.info("Reading configuration YML file...")
        config = load_config(args.config)
        #seed_all(config.featuriser.seed)
        split_dict = torch.load(config.dataset.split)
        logger.info(f"Found {len(split_dict['train'])} samples in the Crossdock dataset for training.")
    
    # Docking ligands with Autodock Vina implemented in ODDT
    logger.info("Extracting features...")
    for i in tqdm(range(len(split_dict['train']))):
        #for i in tqdm(range(1)):
        logger.info(" ")
        complexDict, skippedComplex = {}, []
        name = str(split_dict['train'][i][0].split(".")[0])
        
        outDirExists = os.path.exists(os.path.join(args.outdir, name.split("/")[0]))
        if not outDirExists:
           os.makedirs(os.path.join(args.outdir, name.split('/')[0]))
           logger.info(f"Output directory {name.split('/')[0]} is created.")
        
        logger.info(f"Now reading {name}...")
        proteinDual = StructureDual(os.path.join(config.dataset.path, split_dict['train'][i][0]), isProtein=True)
        ligandDual = StructureDual(os.path.join(config.dataset.path, split_dict['train'][i][1]), isProtein=False)
        
        protein, _protein = proteinDual.parse_to_oddt(), proteinDual.parse_to_rdkit()
        ligand, _ligand = ligandDual.parse_to_oddt(), ligandDual.parse_to_rdkit()
        ligand_com = parse_sdf_file(os.path.join(config.dataset.path, split_dict['train'][i][1]))['center_of_mass']
        
        vina = autodock_vina(protein=protein, 
                             auto_ligand=ligand,
                             center=tuple(ligand_com),
                             num_modes=int(config.autodock.num_modes), 
                             executable=config.autodock.executable)
        
        # Extracting vina score of native structure
        vina_score = float(vina.predict_ligand(ligand).data['vina_affinity'])
        protein_cc, ligand_cc = PLInteraction.close_contacts(x=protein.atom_dict, 
                                                             y=ligand.atom_dict, 
                                                             cutoff=float(config.featuriser.sasa_cutoff))
        protein_cc_idx, ligand_cc_idx = list(np.unique(protein_cc['id'])), list(np.unique(ligand_cc['id']))
        protein_cc_radii, ligand_cc_radii = ClassifyAtoms(config.featuriser.symbol_radius_path, _protein, protein_cc_idx), ClassifyAtoms(config.featuriser.symbol_radius_path, _ligand, ligand_cc_idx)
        
        
        logger.info("Extracting protein node features...")
        # Computing SASA
        try:            
            protein_cc_coords = proteinDual.RetrieveCoords(protein_cc_idx)
            protein_cc_coords = list(itertools.chain.from_iterable(protein_cc_coords))
            
            sasa = ComputeSASA(protein_cc_coords, protein_cc_idx)
            protein_feature = RetrieveMolecularFeatures(molList=[_protein], labelList=[[0.0, vina_score, sasa]])[0]
        except Exception as e:
            logger.error(traceback.format_exc())
            skippedComplex.append(name)
            continue


        logging.info("Extracting ligand node features...")
        # Computing SASA
        try:            
            ligand_cc_coords = ligandDual.RetrieveCoords(ligand_cc_idx)
            ligand_cc_coords = list(itertools.chain.from_iterable(ligand_cc_coords))
            
            sasa = ComputeSASA(ligand_cc_coords, ligand_cc_idx)
            ligand_feature = RetrieveMolecularFeatures(molList=[_ligand], labelList=[[0.0, vina_score, sasa]])[0]
        except Exception as e:
            logger.error(traceback.format_exc())
            skippedComplex.append(name)
            continue
        else:
            logger.info("Retrieving PL interaction (edge) features...")
            fp, fl = PLInteraction.fingerprint(protein, ligand)

            logger.info("Dumping features into pickle...")
            complexDict['name'] = name
            complexDict['vina_score'] = vina_score
            complexDict['protein_feature'] = protein_feature
            complexDict['ligand_feature'] = ligand_feature
            complexDict['interaction_fp'] = fp
            complexDict['interaction_fl'] = fl

            # Pickle dump
            with open(args.outdir + '/' + name + '.pkl', 'wb') as file:
                pickle.dump(complexDict, file)
            """
            with open('trail.pkl', 'wb') as file:
                pickle.dump(complexDict, file)            
            """
    
    logger.info(f"Skipped {len(skippedComplex)} complexes: {skippedComplex}")
    logger.info("Process terminated successfully.\n")
    