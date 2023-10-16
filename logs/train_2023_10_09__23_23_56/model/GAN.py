import torch
import torch.nn as nn

try:
    from model.Embedding import EquivariantEmbedding
    from model.CProMG import Transformer, lap_pe
except:
    from Embedding import EquivariantEmbedding
    from CProMG import Transformer, lap_pe
    
    
class SINGA(nn.Module):
    def __init__(self, config, device='cuda'):
        super().__init__()
        self.device = device
        self.config = config
        self.embedding = EquivariantEmbedding(config=self.config.embedding, device=self.device)
        self.model = Transformer(
            config = self.config.model, 
            protein_atom_feature_dim = self.config.model.featurizer_feat_dim, 
            num_props = self.config.train.num_props, 
            device = self.device,
        )

    def forward(self, g):
        dic = {
            'sas': g['ligand_data']['sas'],
            'logP': g['ligand_data']['logP'],
            'qed': g['ligand_data']['qed'],
            'weight': g['ligand_data']['weight'],
            'tpsa': g['ligand_data']['tpsa'],
            'vina_score': g['ligand_data']['vina_score'],
        }
        
        
        # Prop preparation
        if self.config.train.num_props:
            dic['vina_score'] = (torch.lt(dic['vina_score'], -7.5)).float()
            dic['qed'] = (torch.gt(dic['qed'], 0.6)).float()
            dic['sas'] = (torch.lt(dic['sas'], 4.0)).float()
            props = self.config.train.prop
            prop = torch.tensor(list(zip(*[dic[p] for p in props]))).to(self.device)
        else:
            prop = None


        # Batch preparation    
        _batch, _batch_aa, batch, batch_aa = g['protein_atoms']['ptr'], g['ligand_atoms']['ptr'], list(), list()
        for i in range(_batch.shape[0]-1):
            batch.extend([i] * (_batch[i+1]-_batch[i]))
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)
        
        for i in range(_batch_aa.shape[0]-1):
            batch_aa.extend([i] * (_batch_aa[i+1]-_batch_aa[i]))
        batch_aa = torch.tensor(batch_aa, dtype=torch.long, device=self.device)
        
        
        # Pass to embedding
        embed = self.embedding(g)
        for i in list(embed.keys()):
            embed[i] = embed[i].embedding
        embed['protein'] = (embed['protein_atoms'] + embed['lp_edge']).view(-1, self.config.model.featurizer_feat_dim).to(torch.device(self.device))
        embed['ligand']  = (embed['ligand_atoms']  + embed['pl_edge']).view(-1, self.config.model.featurizer_feat_dim).to(torch.device(self.device))
        
        
        # Pass to CProMG encoder-decoder model
        embed = self.model(
            node_attr = embed['protein'],
            pos = g['protein_atoms']['pos'],
            batch = batch,
            atom_laplacian = lap_pe(data=g, node_type='protein_atoms'),
            smiles_index = g['ligand_data']['smiIndices_input'],
            tgt_len = self.config.model.decoder.tgt_len,
            aa_node_attr = embed['ligand'],
            aa_pos = g['ligand_atoms']['pos'],
            aa_batch = batch_aa,
            aa_laplacian = lap_pe(data=g, node_type='ligand_atoms'),
            prop = prop,
        )
        
        return embed  # output has shape [tgt_len x N, 116], where N is the num of samples
