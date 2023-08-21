import logging
import multiprocessing as mp

import torch
from torch.nn import Linear
import pytorch_lightning as pl

from model import HG_model
from utils.Data import CrossdockedDataModule
from utils.misc import *



# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    #torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
else:
    device = torch.device('cpu')


# Load config file
config = load_config("./config/train.yml")
split_dict = torch.load(config.dataset.split)


# Load data
datamodule = CrossdockedDataModule(root=config.dataset.path,
                                   index=config.dataset.split,
                                   atomic_distance_cutoff=config.embedding.atomic_distance_cutoff,
                                   batch_size=config.embedding.batch_size,
                                   num_workers=1)
datamodule.setup()
datamodule = datamodule.train_dataloader()


# Load model
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


# Show model params
nb_param_trainable = model.get_nb_parameters(only_trainable=True)
nb_param = model.get_nb_parameters(only_trainable=False)
print(f"Total params: {nb_param}; \t Trainable params: {nb_param_trainable}")


# Load some example for testing
data1 = torch.load('./example/1k1j_1yp9.pt').to(torch.device('cuda:0'))
data2 = torch.load('./example/4xe6_3fqc.pt').to(torch.device('cuda:0'))
data3 = torch.load('./example/5ai4_5am0.pt').to(torch.device('cuda:0'))

data1_x_src, data1_x_dst = data1.x_dict.items()
                       