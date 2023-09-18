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
from model.Layers import GATE_GRUConv_IntraMol
from model.P2M_invariant import GVPerceptronVN, MessageModule, GaussianSmearing, EdgeExpansion
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


# Load config file
config = load_config("./config/train.yml")
split_dict = torch.load(config.dataset.split)

# Load data (PASS)
datamodule = CrossdockedDataModule(root=config.dataset.path,
                                   index=config.dataset.split,
                                   atomic_distance_cutoff=config.embedding.atomic_distance_cutoff,
                                   batch_size=config.embedding.batch_size,
                                   num_workers=1)
datamodule.setup()
datamodule = datamodule.train_dataloader()

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
      assert y_pred.shape == (config.embedding.batch_size, \
                              config.embedding.mlp_channels[-1], \
                              (config.embedding.hidden_channels_pa[-1] + config.embedding.hidden_channels_la[-1]) )


# For backup testing (PASS)
protein = next(oddt.toolkit.readfile('pdb', 'example/7cff_protein.pdb'))
protein.protein = True
ligand = next(oddt.toolkit.readfile('sdf', 'example/7cff_ligand.sdf'))


# Test subgraph (PASS)
G_exp_file = ['./dataset/crossdocked_graph10/P53_HUMAN_94_306_0/4agq_B_rec_5a7b_kmn_lig_tt_docked_2_pocket10.pt', 
              './dataset/crossdocked_graph10/PDE10_HUMAN_439_773_0/3wi2_A_rec_4tpp_35d_lig_tt_docked_1_pocket10.pt', 
              './dataset/crossdocked_graph10/BRD4_HUMAN_42_168_0/5cp5_A_rec_4nue_nue_lig_tt_min_0_pocket10.pt'] 
G_exp = []
for i in G_exp_file:
    G_exp.append(torch.load(i).to(device))

ligand_masking = LigandMasking(HeteroData=G_exp[-1], device=device)
masked_idx, content_idx = ligand_masking()
subset_dict = {'ligand_atoms': content_idx}
G_mod = ligand_masking.subset_subgraph(subset_dict=subset_dict)



# Test new layers
ligand_ei = G_exp[-1][('ligand_atoms', 'linked_to', 'ligand_atoms')]['edge_index'].to(device)
ligand_ea = G_exp[-1][('ligand_atoms', 'linked_to', 'ligand_atoms')]['edge_attr'].to(device)
ligand_x = G_exp[-1]['ligand_atoms']['x'].to(device)
ligand_p = G_exp[-1]['ligand_atoms']['pos'].to(device)
ligand_ev = ligand_p[ligand_ei[0]] - ligand_p[ligand_ei[1]]
ligand_ed = torch.norm(ligand_ev, dim=-1, p=2)

scalar, vector = ligand_x, ligand_p.view(-1, 1, 3)
row, col = ligand_ei
edge_channels, num_edge_types = 64, 6


per1 = GVPerceptronVN(59, 1, 32, 1)
gate1 = GATE_GRUConv_IntraMol(in_channels=59, out_channels=32, dropout=0.0, edge_dim=6, dim_hid=1)
msg1 = MessageModule(node_sca=59, node_vec=1, edge_sca=6, edge_vec=1, out_sca=59, out_vec=1)
distance_expansion = GaussianSmearing(stop=10., num_gaussians=6, device=device)
#distance_expansion = GaussianSmearing(stop=10., num_gaussians=edge_channels-num_edge_types, device=device)
vector_expansion = EdgeExpansion(edge_channels, device=device)

edge_sca_feat = torch.cat([distance_expansion(ligand_ed), ligand_ea], dim=-1)
edge_vec_feat = vector_expansion(ligand_ev)

out_per1 = per1(x=ligand_x, pos=ligand_p.view(-1, 1, 3))
out_gate1 = gate1(x=ligand_x, pos=ligand_p, edge_index=ligand_ei, edge_attr=ligand_ea)
msg_j_sca, msg_j_vec = msg1(scalar, vector, edge_sca_feat, edge_vec_feat, col, ligand_ed, annealing=True)

