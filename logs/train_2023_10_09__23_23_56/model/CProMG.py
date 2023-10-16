import math
import numpy as np
from typing import Union, Any
import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, Conv1d, LayerNorm, ReLU, BatchNorm1d, Softmax, Embedding, Dropout

import torch_geometric
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import knn_graph
from torch_geometric.utils import get_laplacian, to_dense_batch, to_undirected
from torch_scatter import scatter_sum, scatter_softmax
import torch_cluster


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_channels, edge_channels, key_channels, num_heads=1, device='cuda'):
        super(MultiHeadAttention, self).__init__()
        assert hidden_channels % num_heads == 0
        assert key_channels % num_heads == 0

        self.device = device
        self.num_heads = num_heads
        self.k_lin = Conv1d(hidden_channels, key_channels, 1, groups=num_heads, bias=False, device=device)
        self.q_lin = Conv1d(hidden_channels, key_channels, 1, groups=num_heads, bias=False, device=device)
        self.v_lin = Conv1d(hidden_channels, hidden_channels, 1, groups=num_heads, bias=False, device=device)

        self.weight_k_net = Sequential(
            Linear(edge_channels, key_channels//num_heads, device=device),
            ShiftedSoftplus(device=device),
            Linear(key_channels//num_heads, key_channels//num_heads, device=device),
        )
        self.weight_k_lin = Linear(key_channels//num_heads, key_channels//num_heads, device=device)

        self.weight_v_net = Sequential(
            Linear(edge_channels, hidden_channels//num_heads, device=device),
            ShiftedSoftplus(device=device),
            Linear(hidden_channels//num_heads, hidden_channels//num_heads, device=device),
        )
        self.weight_v_lin = Linear(hidden_channels//num_heads, hidden_channels//num_heads, device=device)
        self.centroid_lin = Linear(hidden_channels, hidden_channels, device=device)
        self.act = ShiftedSoftplus(device=device)
        self.out_transform = Linear(hidden_channels, hidden_channels, device=device)
        self.layer_norm = LayerNorm(hidden_channels, device=device)
        # self.batch_norm = BatchNorm1d(hidden_channels, device=device)

    def forward(self, node_attr, edge_index, edge_attr):
        N = node_attr.size(0)
        row, col = edge_index

        # Project to multiple key, query and value spaces
        h_keys = self.k_lin(node_attr.unsqueeze(-1)).view(N, self.num_heads, -1)
        h_queries = self.q_lin(node_attr.unsqueeze(-1)).view(N, self.num_heads, -1)
        h_values = self.v_lin(node_attr.unsqueeze(-1)).view(N, self.num_heads, -1)

        # Compute keys and queries
        W_k = self.weight_k_net(edge_attr)
        keys_j = self.weight_k_lin(W_k.unsqueeze(1) * h_keys[col])
        queries_i = h_queries[row]

        # Compute attention weights (alphas)
        qk_ij = ((queries_i * keys_j).sum(-1))/ np.sqrt(keys_j.size(-1))
        alpha = scatter_softmax(qk_ij, row, dim=0)

        # Compose messages
        W_v = self.weight_v_net(edge_attr)  # (E, H_per_head)
        msg_j = self.weight_v_lin(W_v.unsqueeze(1) * h_values[col])
        msg_j = alpha.unsqueeze(-1) * msg_j

         # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N).view(N, -1)
        out = self.centroid_lin(node_attr) + aggr_msg
        out = self.out_transform(self.act(out))
        
        return self.layer_norm(out)


class MultiHeadAttention2(nn.Module):
    def __init__(self, hidden_channels, key_channels, num_heads, device='cuda'):
        super(MultiHeadAttention2, self).__init__()
        self.device = device
        self.hidden_channls = hidden_channels
        self.keys_channels = key_channels
        self.num_heads = num_heads
        self.W_Q = Linear(hidden_channels, key_channels, device=device)
        self.W_K = Linear(hidden_channels, key_channels, device=device)
        self.W_V = Linear(hidden_channels, hidden_channels, device=device)
        self.linear = Linear(hidden_channels, hidden_channels, device=device)
        self.layer_norm = LayerNorm(hidden_channels, device=device)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.keys_channels//self.num_heads).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.num_heads, self.keys_channels//self.num_heads).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.num_heads, self.hidden_channls//self.num_heads).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_channls)
        output = self.linear(context)

        return self.layer_norm(output+residual)
        
        
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class ScaledDotProductDeAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductDeAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        hidden_channels = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(hidden_channels)
        scores.masked_fill_(attn_mask, -1e9)
        attn = Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadDeAttention(nn.Module):
    def __init__(self, hidden_channels, key_channels, num_heads, device='cuda'):
        super(MultiHeadDeAttention, self).__init__()
        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.W_Q = Linear(hidden_channels, key_channels, device=device)
        self.W_K = Linear(hidden_channels, key_channels, device=device)
        self.W_V = Linear(hidden_channels, hidden_channels, device=device)
        self.linear = Linear(hidden_channels, hidden_channels, device=device)
        self.layer_norm = LayerNorm(hidden_channels, device=device)
        self.device = device

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.key_channels // self.num_heads).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.num_heads, self.key_channels // self.num_heads).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.num_heads, self.hidden_channels // self.num_heads).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        context = ScaledDotProductDeAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_channels)
        output = self.linear(context)
        return self.layer_norm(output + residual)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, hidden_channels, device='cuda'):
        super(PoswiseFeedForwardNet, self).__init__()
        self.device = device
        self.conv1 = Conv1d(in_channels=hidden_channels, out_channels=1024, kernel_size=1, device=device)
        self.conv2 = Conv1d(in_channels=1024, out_channels=hidden_channels, kernel_size=1, device=device)
        self.layer_norm = LayerNorm(hidden_channels, device=device)
        self.batch_norm = BatchNorm1d(hidden_channels, device=device)

    def forward(self, inputs):
        residual = inputs
        inputs = inputs.unsqueeze(0)
        output = ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        output = output.squeeze(0)
        return self.layer_norm(output + residual)


class PoswiseFeedForwardDeNet(nn.Module):
    def __init__(self, hidden_channels, device='cuda'):
        super(PoswiseFeedForwardDeNet, self).__init__()
        self.device = device
        self.conv1 = Conv1d(in_channels=hidden_channels, out_channels=1024, kernel_size=1, device=device)
        self.conv2 = Conv1d(in_channels=1024, out_channels=hidden_channels, kernel_size=1, device=device)
        self.layer_norm = LayerNorm(hidden_channels, device=device)

    def forward(self, inputs):
        residual = inputs
        output = ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device='cuda'):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, config, device='cuda'):
        super(EncoderLayer, self).__init__()
        self.config = config
        self.device = device
        self.enc_self_attn = MultiHeadAttention(
            hidden_channels=config.hidden_channels,
            edge_channels=config.edge_channels,
            key_channels=config.key_channels,
            num_heads=config.num_heads,
            device=device,
        )
        self.pos_ffn = PoswiseFeedForwardNet(hidden_channels=config.hidden_channels, device=device)

    def forward(self, node_attr, edge_index, edge_attr):
        msa_outputs = self.enc_self_attn(node_attr, edge_index, edge_attr)
        fnn_outputs = self.pos_ffn(msa_outputs)
        return msa_outputs, fnn_outputs


class EncoderLayer2(nn.Module):
    def __init__(self, config, device='cuda'):
        super(EncoderLayer2, self).__init__()
        self.device = device
        self.config = config
        self.enc_self_attn = MultiHeadAttention(
            hidden_channels=config.hidden_channels,
            edge_channels=config.edge_channels,
            key_channels=config.key_channels,
            num_heads=config.num_heads,
            device=device,
        )
        self.proj = Linear(config.hidden_channels, config.hidden_channels, device=device)
        self.cross_attn = MultiHeadAttention2(
            hidden_channels=config.hidden_channels,
            key_channels=config.key_channels,
            num_heads=config.num_heads,
            device=device,
        )
        self.layer_norm = LayerNorm(config.hidden_channels, device=device)
        self.pos_ffn = PoswiseFeedForwardNet(hidden_channels=config.hidden_channels, device=device)

    def forward(self, node_attr, edge_index, edge_attr, idx, atom_msa_outputs, atom_mask, batch):
        msa_outputs = self.enc_self_attn(node_attr, edge_index, edge_attr)
        
        if idx == 2 or idx == 5:
            atom_msa_output = self.proj(atom_msa_outputs[idx])
            msa_outputs1, msa_outputs1_mask = to_dense_batch(msa_outputs, batch)
            cross_outputs1 = self.cross_attn(msa_outputs1, atom_msa_output, atom_msa_output, atom_mask)
            cross_outputs = torch.masked_select(
                cross_outputs1.view(-1, cross_outputs1.size(-1)),
                msa_outputs1_mask.view(-1).unsqueeze(-1),
            ).view(-1, cross_outputs1.size(-1))
            msa_outputs = self.layer_norm(msa_outputs + cross_outputs)
        
        return self.pos_ffn(msa_outputs)
        

# Atom Encoder
class Encoder(nn.Module):
    def __init__(self, config, protein_atom_feature_dim, device='cuda'):
        super(Encoder, self).__init__()
        self.config = config
        self.device = device
        self.protein_atom_feature_dim = protein_atom_feature_dim
        self.protein_atom_emb = Linear(protein_atom_feature_dim, config.hidden_channels, device=device)
        self.laplacian_emb = Linear(config.lap_dim, config.hidden_channels, device=device)
        self.layers = nn.ModuleList([EncoderLayer(config, device=device) for _ in range(config.num_interactions)])
        self.distance_expansion = GaussianSmearing(stop=15, num_gaussians=config.edge_channels, device=device)
        self.out = Linear(config.hidden_channels, config.hidden_channels, device=device)
        self.layer_norm = LayerNorm(config.hidden_channels, device=device)

    def forward(self, protein_atom_feature, pos, batch, atom_laplacian):
        node_attr = self.protein_atom_emb(protein_atom_feature)
        atom_laplacian = self.laplacian_emb(atom_laplacian)
        node_attr = node_attr + atom_laplacian
        edge_index_di = knn_graph(pos, self.config.knn, batch=batch, flow='target_to_source')
        # edge_index = radius_graph(pos,4.5 ,batch=batch, flow='target_to_source')
        edge_length = torch.norm(pos[edge_index_di[0]] - pos[edge_index_di[1]], dim=1)
        edge_index, edge_attr = to_undirected(edge_index_di, edge_length, reduce='mean')
        edge_attr = self.distance_expansion(edge_attr)
        edge_index, edge_attr = get_laplacian(edge_index, edge_attr)

        msa_outputs1 = []
        for layer in self.layers:
            msa_outputs, fnn_outputs = layer(node_attr, edge_index, edge_attr)
            node_attr = fnn_outputs
            msa_outputs, msa_pad_mask = to_dense_batch(msa_outputs, batch)
            msa_outputs1.append(msa_outputs)
        enc_outputs1, enc_pad_mask1 = to_dense_batch(node_attr, batch)
        enc_pad_mask1 = ~enc_pad_mask1.unsqueeze(1)

        return enc_outputs1, enc_pad_mask1, msa_outputs1


# AA Encoder
class Encoder2(nn.Module):
    def __init__(self, config, aa_feature_dim, device='cuda'):
        super(Encoder2, self).__init__()
        self.device = device
        self.config = config
        self.aa_feature_dim = aa_feature_dim
        self.aa_emb = Linear(aa_feature_dim, config.hidden_channels, device=self.device)
        self.laplacian_emb = Linear(config.lap_dim, config.hidden_channels, device=self.device)
        self.layers = nn.ModuleList([EncoderLayer2(config, device=self.device) for _ in range(config.num_interactions)])
        self.distance_expansion = GaussianSmearing(stop=25, num_gaussians=config.edge_channels, device=self.device)
        self.out = Linear(config.hidden_channels, config.hidden_channels, device=self.device)
        self.layer_norm = LayerNorm(config.hidden_channels, device=self.device)

    def forward(self, aa_feature, aa_pos, aa_batch, aa_laplacian, atom_mask, atom_msa_outputs):
        node_attr = self.aa_emb(aa_feature)
        aa_laplacian = self.laplacian_emb(aa_laplacian)
        node_attr = node_attr + aa_laplacian
        edge_index = knn_graph(aa_pos, 30, batch=aa_batch, flow='target_to_source')
        edge_length = torch.norm(aa_pos[edge_index[0]] - aa_pos[edge_index[1]], dim=1)
        edge_index, edge_attr = to_undirected(edge_index, edge_length, reduce='mean')
        edge_attr = self.distance_expansion(edge_attr)
        edge_index, edge_attr = get_laplacian(edge_index, edge_attr)

        for idx, layer in enumerate(self.layers):
            enc_outputs = layer(node_attr, edge_index, edge_attr, idx, atom_msa_outputs, atom_mask, aa_batch)
            node_attr = enc_outputs

        enc_outputs1, enc_pad_mask1 = to_dense_batch(node_attr, aa_batch)
        enc_pad_mask1 = ~enc_pad_mask1.unsqueeze(1)

        return enc_outputs1, enc_pad_mask1


class DecoderLayer(nn.Module):
    def __init__(self, config, device='cuda'):
        super(DecoderLayer, self).__init__()
        self.device = device
        self.dec_self_attn = MultiHeadDeAttention(
            hidden_channels=config.hidden_channels,
            key_channels=config.key_channels,
            num_heads=config.num_heads,
            device=device,
        )
        self.dec_enc_attn = MultiHeadDeAttention(
            hidden_channels=config.hidden_channels,
            key_channels=config.key_channels,
            num_heads=config.num_heads,
            device=device,
        )
        self.pos_ffn = PoswiseFeedForwardDeNet(hidden_channels=config.hidden_channels, device=device)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs


class Decoder(nn.Module):
    def __init__(self, config, num_props=None, device='cuda'):
        super(Decoder, self).__init__()
        self.config = config
        self.device = device
        self.num_props = num_props
        self.mol_emb = Embedding(len(config.smiVoc), config.hidden_channels, 0, device=device)
        self.pos_emb = PositionalEncoding(config.hidden_channels, device=device)
        self.type_emb = Embedding(2, config.hidden_channels, device=device)
        if self.num_props:
            self.prop_nn = Linear(self.num_props, config.hidden_channels, device=device)

        self.layers = nn.ModuleList([DecoderLayer(config, device=device) for _ in range(config.num_interactions)])

    def forward(self, smiles_index, enc_outputs, enc_pad_mask, tgt_len, prop=None):
        b, t = smiles_index.size()
        dec_inputs = self.mol_emb(smiles_index)
        dec_inputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1)

        if self.num_props:
            assert prop.shape[-1] == self.num_props
            type_embeddings = self.type_emb(torch.ones((b, t), dtype=torch.long, device=self.device))
            dec_inputs = dec_inputs + type_embeddings
            type_embd = self.type_emb(torch.zeros((b, 1), dtype=torch.long, device=self.device))
            p = self.prop_nn(prop.unsqueeze(1))
            p += type_embd
            dec_inputs = torch.cat([p, dec_inputs], 1)
            con = torch.ones(smiles_index.shape[0], 1, device=self.device)
            smiles_index = torch.cat([con, smiles_index], 1)

        if self.num_props:
            num = int(bool(self.num_props))
        else:
            num = 0

        dec_self_attn_pad_mask = get_attn_pad_mask(
            smiles_index,
            smiles_index,
            self.config.smiVoc.index('^'),
        )
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(smiles_index)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = enc_pad_mask.expand(
            enc_pad_mask.size(0),
            tgt_len + num,
            enc_pad_mask.size(2),
        )  # batch_size x len_q x len_k

        for layer in self.layers:
            dec_outputs = layer(dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_inputs = dec_outputs

        return dec_outputs


class Transformer(nn.Module):
    def __init__(self, config, protein_atom_feature_dim, num_props=None, device='cuda'):
        super(Transformer, self).__init__()
        self.config = config
        self.num_props = num_props
        self.encoder = Encoder(config.encoder, protein_atom_feature_dim, device=device)
        self.encoder2 = Encoder2(config.encoder, protein_atom_feature_dim, device=device)
        self.decoder = Decoder(config.decoder, self.num_props, device=device)
        self.projection = nn.Linear(config.hidden_channels, len(config.decoder.smiVoc), bias=False, device=device)
        self.device = device
        
    def forward(
        self,
        node_attr,
        pos,
        batch,
        atom_laplacian,
        smiles_index,
        tgt_len,
        aa_node_attr,
        aa_pos,
        aa_batch,
        aa_laplacian,
        prop=None,
    ):
        enc_outputs1, enc_pad_mask1, msa_outputs = self.encoder(node_attr, pos, batch, atom_laplacian)
        enc_outputs2, enc_pad_mask2 = self.encoder2(aa_node_attr, aa_pos, aa_batch, aa_laplacian, enc_pad_mask1, msa_outputs)
        enc_outputs = torch.cat([enc_outputs1, enc_outputs2], dim=1)
        enc_pad_mask = torch.cat([enc_pad_mask1, enc_pad_mask2], dim=2)
        dec_outputs = self.decoder(smiles_index, enc_outputs, enc_pad_mask, tgt_len,prop)  # dec_outputs : [num_samples, tgt_len+1, 256]
        dec_logits = self.projection(dec_outputs)  # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]

        if self.num_props:
            num = int(bool(self.num_props))
        else:
            num = 0

        dec_logits = dec_logits[:, num:, :]
        return dec_logits.reshape(-1, dec_logits.size(-1))


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50, device='cuda'):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians, device=device)
        self.coeff = -0.5 / (offset[1]-offset[0]).item()**2
        self.register_buffer('offset', offset)
        self.device = device

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.shift = torch.log(torch.tensor(2.0, device=device)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


def get_attn_pad_mask(seq_q, seq_k, pad_id):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(pad_id).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


def to_dgl(data: Union['torch_geometric.data.Data', 'torch_geometric.data.HeteroData']) -> Any:
    if isinstance(data, Data):
        if data.edge_index is not None:
            row, col = data.edge_index
        else:
            row, col, _ = data.adj_t.t().coo()

        g = dgl.graph((row, col))

        for attr in data.node_attrs():
            g.ndata[attr] = data[attr]
        for attr in data.edge_attrs():
            if attr in ['edge_index', 'adj_t']:
                continue
            g.edata[attr] = data[attr]

        return g

    if isinstance(data, HeteroData):
        data_dict = {}
        for edge_type, store in data.edge_items():
            if store.get('edge_index') is not None:
                row, col = store.edge_index
            else:
                row, col, _ = store['adj_t'].t().coo()

            data_dict[edge_type] = (row, col)

        g = dgl.heterograph(data_dict)

        for node_type, store in data.node_items():
            for attr, value in store.items():
                g.nodes[node_type].data[attr] = value

        for edge_type, store in data.edge_items():
            for attr, value in store.items():
                if attr in ['edge_index', 'adj_t']:
                    continue
                g.edges[edge_type].data[attr] = value

        return g

    raise ValueError(f"Invalid data type (got '{type(data)}')")
    

def lap_pe(data, node_type:str):
    assert node_type in ['protein_atoms', 'ligand_atoms'], "Node type not accepted"
    homo_D = Data(x=data[node_type]['x'], 
                  pos=data[node_type]['pos'], 
                  edge_index=data[(node_type, 'linked_to', node_type)]['edge_index'],
                  edge_attr=data[(node_type, 'linked_to', node_type)]['edge_attr'])
    homo_g = to_dgl(homo_D)
    laplacian = dgl.lap_pe(homo_g, 8)
    
    return laplacian
