import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import logging
import math
import multiprocessing as mp
import oddt
import numpy as np

import torch
from torch import Tensor
from torch.nn import Linear
import torch_geometric as pyg
from torch_sparse import SparseTensor
import pytorch_lightning as pl

from model import HG_model
from model.Masking import LigandMasking
from model.Embedding import EquivariantEmbedding
from utils.Data import CrossdockedDataModule
from utils.misc import *


def print_layer(prefix):
    print(f'{prefix}: {os.environ.get("MKL_THREADING_LAYER")}')

def child():
    import torch
    torch.set_num_threads(1)

if __name__ == '__main__':
    import mkl
    from torch import multiprocessing as mp

    mp.set_start_method('spawn')
    p = mp.Process(target=child)
    p.start()
    p.join()
    

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    #torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn') # good solution !!!!
else:
    device = torch.device('cpu')


device = torch.device('cpu')

# Load config file
config = load_config("./config/train.yml")
split_dict = torch.load(config.dataset.split)

# Load data (PASS)
datamodule = CrossdockedDataModule(root=config.dataset.path,
                                   index=config.dataset.split,
                                   atomic_distance_cutoff=config.dataloader.atomic_distance_cutoff,
                                   batch_size=config.dataloader.batch_size,
                                   num_workers=config.dataloader.num_workers,
                                   device=device)
datamodule.setup()
# datamodule = datamodule.train_dataloader()

"""
# Load model (PASS)
model = HG_model.HG_Model(hidden_channels_pa=config.embedding.hidden_channels_pa,
                          hidden_channels_la=config.embedding.hidden_channels_la,
                          num_layers=config.embedding.num_layers,
                          dropout=config.embedding.dropout,
                          heads=config.embedding.heads,
                          hetero_aggr=config.embedding.hetero_aggr,
                          mlp_channels=config.embedding.mlp_channels,
                          lr=config.embedding.lr,
                          weight_decay=config.embedding.weight_decay,
                          molecular_embedding_size=config.embedding.molecular_embedding_size,
                          str_for_hparams="InterMol length: {}".format(config.embedding.atomic_distance_cutoff),
                          )
if use_cuda:
    model = model.to(torch.device('cuda:0'))
print(model)

# Show model params (PASS)
nb_param_trainable = model.get_nb_parameters(only_trainable=True)
nb_param = model.get_nb_parameters(only_trainable=False)
print(f"Total params: {nb_param}; \t Trainable params: {nb_param_trainable}")

# Load some example for testing (PASS)
for i, X in enumerate(datamodule):
    if i == 0:
      x_batch = X
      y_pred = model(X)
"""

# For backup testing (PASS)
protein = next(oddt.toolkit.readfile('pdb', 'example/7cff_protein.pdb'))
protein.protein = True
ligand = next(oddt.toolkit.readfile('sdf', 'example/7cff_ligand.sdf'))


# Test subgraph (PASS)
G_exp_file = ['./dataset/crossdocked_graph10_v3/P53_HUMAN_94_306_0/4agq_B_rec_5a7b_kmn_lig_tt_docked_2_pocket10.pt', 
              './dataset/crossdocked_graph10_v3/PDE10_HUMAN_439_773_0/3wi2_A_rec_4tpp_35d_lig_tt_docked_1_pocket10.pt', 
              './dataset/crossdocked_graph10_v3/BRD4_HUMAN_42_168_0/5cp5_A_rec_4nue_nue_lig_tt_min_0_pocket10.pt'] 
G_exp = []
for i in G_exp_file:
    G_exp.append(torch.load(i).to(device))

ligand_masking = LigandMasking(HeteroData=G_exp[-1], device=device)
masked_idx, content_idx = ligand_masking()
subset_dict = {'ligand_atoms': content_idx}
G_mod = ligand_masking.subset_subgraph(subset_dict=subset_dict)


# Spare components
ligand_ei = G_exp[-1][('ligand_atoms', 'linked_to', 'ligand_atoms')]['edge_index'].to(device)
ligand_ea = G_exp[-1][('ligand_atoms', 'linked_to', 'ligand_atoms')]['edge_attr'].to(device)
ligand_x = G_exp[-1]['ligand_atoms']['x'].to(device)
ligand_p = G_exp[-1]['ligand_atoms']['pos'].to(device)
ligand_ev = ligand_p[ligand_ei[0]] - ligand_p[ligand_ei[1]]
ligand_ed = torch.norm(ligand_ev, dim=-1, p=2)

scalar, vector = ligand_x, ligand_p.view(-1, 1, 3)
row, col = ligand_ei

"""
# Testing EquiformerV2 equiavriant embedding net
ebd_graph, batch = [], []
embedding = EquivariantEmbedding(config=config.embedding, device=device)

for i, data in enumerate(datamodule):
    if i == 0:
        batch.append(data)
        ebd_graph.append(embedding(data))
    else:
        continue

ebd_graph[0]['protein_atoms'] = ebd_graph[0]['protein_atoms'].embedding
ebd_graph[0]['ligand_atoms'] = ebd_graph[0]['ligand_atoms'].embedding
ebd_graph[0]['pl_edge'] = ebd_graph[0]['pl_edge'].embedding
ebd_graph[0]['lp_edge'] = ebd_graph[0]['lp_edge'].embedding
"""
