import torch
import torch_geometric
from torch_geometric.data import HeteroData

import numpy as np

from tqdm import tqdm
import os
import pickle
import yaml
import random
import warnings
import argparse
import traceback
import logging

from utils.misc import load_config



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='./config/train.yml')
    parser.add_argument('--data', type=str,
                        default='./dataset/crossdocked_vina10')
    parser.add_argument('--device', type=str,
                        default='cpu')
    parser.add_argument('--logdir', type=str,
                        default='./logs')
    parser.add_argument('--outdir', type=str, 
                        default='./output')
    args = parser.parse_args()
    
        
    # Logging
    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))
    
    # Load config
    outDirExists = os.path.isfile(args.config)
    if not outDirExists:
        raise FileNotFoundError("Configuration YML (config.yml) file does not exist or is not specified.")
    else:
        logger.info("Reading configuration YML file...")
        config = load_config(args.config)
        seed_all(config.featuriser.seed)
        split_dict = torch.load(config.dataset.split)
        logger.info(f"Found {len(split_dict['train'])} samples in the Crossdock dataset for training.")
        if config.train.use_apex:
            from apex import amp
        
        
    print("\n")
    outDirExists = os.path.exists(args.outdir)
    if not outDirExists:
       os.makedirs(args.outdir)
       logger.info("Output directory {args.outdir} is created.")





    # Model
    logger.info('Building model...')
    if config.model.vn == 'singa':
        model = MaskFillModelVN(
            config.model, 
            num_classes = contrastive_sampler.num_elements,
            num_bond_types = edge_sampler.num_bond_types,
            protein_atom_feature_dim = protein_featurizer.feature_dim,
            ligand_atom_feature_dim = ligand_featurizer.feature_dim,
        ).to(args.device)
    print('Num of parameters is', np.sum([p.numel() for p in model.parameters()]))

    # Optimizer and scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    if config.train.use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1'





    try:
        model.train()
        for it in range(1, config.train.max_iters+1):
            try:
                train(it)
            except RuntimeError as e:
                logger.error('Runtime Error ' + str(e))
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
                model.train()
    except KeyboardInterrupt:
        logger.info('Terminating...')    
        
