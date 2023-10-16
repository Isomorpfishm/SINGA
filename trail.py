import re
import pickle
from easydict import EasyDict
import pandas as pd
import dgl
import torch
import torch.nn as nn
from torch_geometric.data import Data
from model import CProMG
from model.BeamSearch import beam_search
from utils.misc import load_config, seed_all

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Load configs
config = './config/train.yml'
config = load_config(config)
seed_all(config.train.seed)

with open('dataloader.pickle', 'rb') as file:
    dataloader = pickle.load(file)
    
with open('embedding.pickle', 'rb') as file:
    embedding = pickle.load(file)
    
embedding['protein'] = (embedding['protein_atoms'] + embedding['lp_edge']).view(-1, 784).to(device)
embedding['ligand'] = (embedding['ligand_atoms'] + embedding['pl_edge']).view(-1, 784).to(device)

encoder = CProMG.Encoder(config=config.model.encoder, protein_atom_feature_dim=784, device=device)
print(encoder)

_batch = dataloader['protein_atoms']['ptr']
batch = list()

for i in range(_batch.shape[0]-1):
    batch.extend([i] * (_batch[i+1]-_batch[i]))    
batch = torch.tensor(batch, dtype=torch.long, device=device)

input_L = torch.rand((embedding['protein'].shape[0], 8), device=device, dtype=torch.float32)
input_T = embedding['protein']
input_pos = torch.rand((embedding['protein'].shape[0], 3), device=device, dtype=torch.float32)
output_T = encoder(input_T, pos=input_pos, batch=batch, atom_laplacian=input_L)
enc_outputs1, enc_pad_mask1, msa_outputs = output_T

smiles_index = dataloader['ligand_data']['smiIndices_input']
enc_outputs, enc_pad_mask = enc_outputs1, enc_pad_mask1
num_props, tgt_len = 3, 200
props = ['vina_score', 'qed', 'sas']

dic = dict()

qed = dataloader['ligand_data']['qed']
dic['qed'] = (torch.gt(qed, 0.6)).float()
vina_score = dataloader['ligand_data']['vina_score']
dic['vina_score'] = (torch.lt(vina_score, 0.6)).float()
sas = dataloader['ligand_data']['sas']
dic['sas'] = (torch.lt(sas, 0.6)).float()

dic['qed'] = torch.unsqueeze(dic['qed'], dim=-1)
dic['sas'] = torch.unsqueeze(dic['sas'], dim=-1)
dic['vina_score'] = torch.unsqueeze(dic['vina_score'], dim=-1)
prop = torch.tensor(list(zip(*[dic[p] for p in props]))).to(device)

decoder =  CProMG.Decoder(config=config.model.decoder, num_props=num_props, device='cpu')
dec_outputs = decoder(smiles_index, enc_outputs, enc_pad_mask, tgt_len=tgt_len, prop=torch.tensor(prop))
print(dec_outputs.shape)

projection = nn.Linear(256, len(config.model.decoder.smiVoc), bias=False)
dec_logits = projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
num = int(bool(num_props))
dec_logits = dec_logits[:, num:, :]
output = dec_logits.reshape(-1, dec_logits.size(-1))
print(output.shape)
