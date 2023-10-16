import os, sys
import argparse
import logging
import mkl
import shutil
import traceback
import warnings
from tqdm import tqdm
from termcolor import colored

import torch
from torch import multiprocessing as mp
from torch.utils import tensorboard
from torch.nn.utils import clip_grad_norm_

from model.CProMG import Transformer
from model.GAN import SINGA
from utils.Data import CrossdockedDataModule
from utils.Stopping import EarlyStopping
from utils.misc import (load_config, 
                        get_new_log_dir, 
                        get_logger, 
                        get_optimizer, 
                        get_scheduler, 
                        seed_all)


def child():
    torch.set_num_threads(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/train.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()
    
    
    # Load config
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)
    
    
    # Logging    
    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('training_log', log_dir)
    logger.info("Process started...")
    logger.info("Reading configuration YML file...")
    logger.info(args)
    logger.info(config)
    
    writer = tensorboard.SummaryWriter(log_dir)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./model', os.path.join(log_dir, 'model'))
    
    
    # Initialise multiprocessing
    mp.set_start_method('spawn')
    p = mp.Process(target=child)
    p.start()
    p.join()

    
    # Use CUDA for PyTorch and PyTorch Geometric
    use_cuda = torch.cuda.is_available()
    if use_cuda and (args.device == 'cuda'):
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        # torch.multiprocessing.set_sharing_strategy('file_system')
        torch.multiprocessing.set_start_method('spawn') # good solution
    else:
        device = torch.device('cpu')
    
    
    # Load and split data
    split_dict = torch.load(config.dataset.split)
    datamodule = CrossdockedDataModule(root=config.dataset.path,
                                       index=config.dataset.split,
                                       atomic_distance_cutoff=config.dataloader.atomic_distance_cutoff,
                                       batch_size=config.dataloader.batch_size,
                                       num_workers=config.dataloader.num_workers,
                                       device=args.device)
    datamodule.setup()
    train_module = datamodule.train_dataloader()
    val_module = datamodule.val_dataloader()
    test_module = datamodule.test_dataloader()

    print(f"Detected {datamodule.train_dataloader().__len__()} batches of training data")
    print(f"Detected {datamodule.val_dataloader().__len__()} batches of validating data")
    print(f"Detected {datamodule.test_dataloader().__len__()} batches of testing data")
    
    
    # Model
    logger.info("Building model...")
    model = SINGA(
        config = config,
        device = args.device,
    )

    # Optimizer and scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    early_stopping = EarlyStopping(mode='min', patience=20, delta=0.00005)


    # Training
    def train(it):
        model.train()
        optimizer.zero_grad()
        batch = next(enumerate(train_module))[1].to(args.device)
        atom_noise = torch.randn_like(batch['protein_atoms']['pos']) * config.train.pos_noise_std 

        outputs = model(
            g = batch,
        )
        
        loss = criterion(outputs, batch['ligand_data']['smiIndices_tgt'].contiguous().view(-1))
        loss.backward()
        
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        del outputs, batch
        
        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad', orig_grad_norm, it)
        writer.flush()


    # Validation
    def validate(it):
        sum_loss, sum_n = 0, 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(enumerate(val_module), desc='Validate'):
                batch = batch[1].to(args.device)
                dic = {
                    'sas': batch['ligand_data']['sas'],
                    'logP': batch['ligand_data']['logP'],
                    'qed': batch['ligand_data']['qed'],
                    'tpsa': batch['ligand_data']['tpsa'],
                    'vina_score': batch['ligand_data']['vina_score'],
                }

                if config.train.num_props:
                    dic['vina_score'] = (torch.lt(dic['vina_score'], -7.5)).float()
                    dic['qed'] = (torch.gt(dic['qed'], 0.6)).float()
                    dic['sas'] = (torch.lt(dic['sas'], 4.0)).float()
                    props = config.train.prop
                    prop = torch.tensor(list(zip(*[dic[p] for p in props]))).to(args.device)
                else:
                    prop = None
                
                outputs = model(
                    g = batch,
                )
                
                loss = criterion(outputs, batch['ligand_data']['smiIndices_tgt'].contiguous().view(-1))
                sum_loss += loss.item()
                sum_n += 1
                
                del outputs, batch
                
        avg_loss = sum_loss / sum_n
        
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()
        
        logger.info(f'[Validate] Iter {it:05d} | Loss {colored(avg_loss, "red")}')
        writer.add_scalar('val/loss', avg_loss, it)
        writer.flush()
        
        return avg_loss


    # Testing
    def test(it):
        sum_loss, sum_n = 0, 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(enumerate(test_module)):
                batch = batch[1].to(args.device)
                dic = {
                    'sas': batch['ligand_data']['sas'],
                    'logP': batch['ligand_data']['logP'],
                    'qed': batch['ligand_data']['qed'],
                    'tpsa': batch['ligand_data']['tpsa'],
                    'vina_score': batch['ligand_data']['vina_score'],
                }
                
                if config.train.num_props:
                    dic['vina_score'] = (torch.lt(dic['vina_score'], -7.5)).float()
                    dic['qed'] = (torch.gt(dic['qed'], 0.6)).float()
                    dic['sas'] = (torch.lt(dic['sas'], 4.0)).float()
                    props = config.train.prop
                    prop = torch.tensor(list(zip(*[dic[p] for p in props]))).to(args.device)
                else:
                    prop = None
                
                outputs = model(
                    g = batch,
                )
                
                loss = criterion(outputs, batch['ligand_data']['smiIndices_tgt'].contiguous().view(-1))
                sum_loss += loss.item()
                sum_n += 1
                
                del outputs, batch
                
        avg_loss = sum_loss / sum_n
        logger.info('[Test] Iter %05d | Loss %.6f' % (it, avg_loss))
        writer.add_scalar('val/loss2', avg_loss, it)
        
        return avg_loss
        

    # Start process
    try:
        for it in range(1, config.train.max_iters+1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                avg_loss = validate(it)
                update, _, counts = early_stopping(avg_loss)
                
                if update:
                    logger.info(colored(f'Update!', 'red'))
                else:
                    logger.info(f'Early stop counter: {counts}/20')
                    
                if early_stopping.early_stop:
                    logger.info(f"{'':12s} Early stop")
                    logger.info(f"{'':->120s}")
                    
                if it > 250000 and it % 10000 == 0:
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                    
                test(it)
    except KeyboardInterrupt:
        logger.info('Terminating ...')
        
    except RuntimeError as e:
        logger.error('Runtime Error ' + str(e))

