import os, sys
import re
import argparse
import logging
import mkl
import shutil
import traceback
import warnings
from easydict import EasyDict
import pandas as pd

import torch
from torch import multiprocessing as mp
from torch.utils import tensorboard

from torch_geometric.data import Data
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.transforms.add_positional_encoding import AddLaplacianEigenvectorPE
from torch_geometric.utils import to_undirected

from model.GAN import SINGA
from model.CProMG import GaussianSmearing, Gaussian, lap_pe
from model.Embedding import EquivariantEmbedding
from model.BeamSearch import beam_search
from utils.PLParser import StructureDual
from utils.Stopping import EarlyStopping
from utils.gen import create_pyg_graph
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
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--model',  type=str, default='./pretrained/SINGA.pt')
    parser.add_argument('--output', type=str, default='./output/result.csv')
    parser.add_argument('--input',  type=str, default='./example/1ifc_A_rec_2ifb_plm_lig_tt_min_0_pocket10.pdb')
    args = parser.parse_args()
    
    
    # Load config
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)
    
    
    # Logging    
    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('generating_log', log_dir)
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
    
    
    # Loading pretrained model
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    it = checkpoint['iteration']
    model.eval()
    
    
    # Make protein graph
    logger.info(f"Reading input protein pocket")
    try:
        proteinDual = StructureDual(args.input, isProtein=True)
        list_atom_name = proteinDual.RetrieveAtomNames()
        name = str(args.input.split("/")[-1].split(".")[0])
        protein = proteinDual.parse_to_oddt()
        
        logger.info("Creating protein pyg graph...")
        g = create_pyg_graph(
            protein = protein, 
            cutoff = config.featuriser.interaction_cutoff, 
            list_atom_name = list_atom_name,
            name = name,
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        sys.exit("Error reading input protein pocket")


    # CProMG method
    atom_laplacian = AddLaplacianEigenvectorPE(k=config.model.encoder.lap_dim, attr_name='protein_atom_laplacian')
    distance_expansion = GaussianSmearing(stop=10.0, num_gaussians=2, device=args.device)
    gaussian = Gaussian(sigma=15)
    
    edge_index = knn_graph(g['protein_atoms']['pos'], 8, flow='target_to_source')
    edge_length = torch.norm(g['protein_atoms']['pos'][edge_index[0]] - g['protein_atoms']['pos'][edge_index[1]], dim=1)
    edge_attr = gaussian(edge_length)
    edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce='mean')
    
    g_homo = Data(x=g['protein_atoms']['x'], 
                  pos=g['protein_atoms']['pos'], 
                  edge_index=g[('protein_atoms', 'linked_to', 'protein_atoms')]['edge_index'],
                  edge_attr=g[('protein_atoms', 'linked_to', 'protein_atoms')]['edge_attr'])
    
    g['protein_atom_laplacian'] = atom_laplacian(data=g_homo)['protein_atom_laplacian']
    g['protein_element_batch'] = torch.zeros([len(g['protein_atoms']['x'])]).long()
    g.to(args.device)
    print(g)
    del(g_homo)
    
    
    # Embedding 
    embedding = EquivariantEmbedding(config=config.embedding, device=args.device)
    embed = embedding(g, gen_mode=True)
    embed['protein_atom_feature'] = embed['protein_atoms'].embedding
    embed['protein_atom_feature'] = embed['protein_atom_feature'].view(-1, config.model.featurizer_feat_dim).to(torch.device(args.device))
    
    
    # Generating
    batch_size = 1
    num_beams = 20
    topk = 1
    filename = g['name']
    
    if config.train.num_props:
        prop = torch.tensor([config.generate.prop for i in range(batch_size*num_beams)], dtype=torch.float, device=args.device)
        assert prop.shape[-1] == config.train.num_props
        num = int(bool(config.train.num_props))
    else:
        num = 0
        prop = None
    
    
    input_data = dict()
    input_data['protein_element_batch'] = g['protein_element_batch']
    input_data['protein_atom_feature'] = embed['protein_atom_feature']
    input_data['protein_pos'] = g['protein_atoms']['pos']
    input_data['protein_atom_laplacian'] = g['protein_atom_laplacian']
    input_data = EasyDict(input_data)
        
    beam_output = beam_search(
        model,
        config.model.decoder.smiVoc,
        num_beams,
        batch_size,
        config.model.decoder.tgt_len + num,
        topk,
        input_data,
        prop,
        device = args.device,
    ).view(batch_size, topk, -1)
    print(beam_output)
    
    
    # Writing results
    for i, item in enumerate(beam_output):
        generate = list()
        for j in item:
            smile = [config.model.decoder.smiVoc[n.item()] for n in j.squeeze()]
            smile = re.sub('[&$^]', '', ''.join(smile))
            generate.append(smile)
            
        logger.info('\n[protein] : %s \n [generate] : %s \n' % (filename[i], generate))
        
        df1 = pd.DataFrame([filename[i]]*topk, columns=['PROTEINS'])
        df2 = pd.DataFrame(generate, columns=['SMILES'])
        df3 = pd.concat([df1, df2], join='outer', axis=1)
    
    # df3.to_csv(args.out, index=False)
