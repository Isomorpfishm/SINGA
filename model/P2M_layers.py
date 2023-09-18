import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import GRUCell, LayerNorm, LeakyReLU
import torch.nn.functional as F

import torch_geometric as pyg
from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_sum

try:
    from P2M_invariant import GVLinear, GVPerceptronVN, MessageModule, EdgeExpansion, GaussianSmearing, VNLeakyReLU
except:
    from model.P2M_invariant import GVLinear, GVPerceptronVN, MessageModule, EdgeExpansion, GaussianSmearing, VNLeakyReLU


### Adapted from https://github.com/KevinCrp/HGScore/blob/main/HGScore/layers/layers.py ###

class GATE_GRUConv_IntraMol(nn.Module):
    """
    A layer gathering a GATEConv and a GRU layer
    First step in atomic embedding for intramolecular network
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        edge_dim: int,
        dim_hid: int = 1,
        cutoff: float = 10.,
        device: str = 'cuda',
    ) -> None:
        """
        Construct a GATE_GRUConv layer

        Args:
            in_channels (int): The input channel size
            out_channels (int): The output channels size
            dropout (float): The dropout rate
            edge_dim (int): The edge dimension
        """
        super().__init__()
        self.dropout = dropout
        self.dim_hid = dim_hid
        self.device = device

        self.per1 = GVPerceptronVN(in_scalar=in_channels, in_vector=dim_hid,
                                   out_scalar=out_channels, out_vector=dim_hid,
                                   device=device)
        self.msg1 = MessageModule(node_sca=out_channels, node_vec=dim_hid,
                                  edge_sca=2*edge_dim, edge_vec=dim_hid,
                                  out_sca=out_channels, out_vec=dim_hid,
                                  cutoff=cutoff, device=device)
        self.gru = GRUCell(out_channels, out_channels, device=device)
        self.centroid_lin = GVLinear(out_channels, dim_hid, out_channels, dim_hid, device=device)
        self.out_transform = GVLinear(out_channels, dim_hid, out_channels, dim_hid, device=device)
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_dim, device=device)
        self.vector_expansion = EdgeExpansion(dim_hid, device=device)
        self.act_sca = LeakyReLU()
        self.act_vec = VNLeakyReLU(1, device=device)
        self.layernorm_sca = LayerNorm([out_channels], device=device)
        self.layernorm_vec = LayerNorm([dim_hid, 3], device=device)

        self.reset_parameters()

    def reset_parameters(self):
        """ GATEConv weights and biases are already initialised """
        glorot(self.per1.lin_vector_weight)
        glorot(self.per1.lin_vector2_weight)
        glorot(self.per1.scalar_to_vector_gates_weight)
        glorot(self.per1.lin_scalar_weight)

        glorot(self.centroid_lin.lin_vector_weight)
        glorot(self.centroid_lin.lin_vector2_weight)
        glorot(self.centroid_lin.scalar_to_vector_gates_weight)
        glorot(self.centroid_lin.lin_scalar_weight)

        glorot(self.out_transform.lin_vector_weight)
        glorot(self.out_transform.lin_vector2_weight)
        glorot(self.out_transform.scalar_to_vector_gates_weight)
        glorot(self.out_transform.lin_scalar_weight)

        zeros(self.per1.lin_vector_bias)
        zeros(self.per1.lin_vector2_bias)
        zeros(self.per1.scalar_to_vector_gates_bias)
        zeros(self.per1.lin_scalar_bias)

        zeros(self.centroid_lin.lin_vector_bias)
        zeros(self.centroid_lin.lin_vector2_bias)
        zeros(self.centroid_lin.scalar_to_vector_gates_bias)
        zeros(self.centroid_lin.lin_scalar_bias)

        zeros(self.out_transform.lin_vector_bias)
        zeros(self.out_transform.lin_vector2_bias)
        zeros(self.out_transform.scalar_to_vector_gates_bias)
        zeros(self.out_transform.lin_scalar_bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        """
        Process data graph through the layer

        Args:
            x (Tensor): Nodes features
            edge_index (Tensor): Edge indices
            edge_attr (Tensor): Edge attributes

        Output:
            Tensor: New node attributes
        """
        x_unpack, pos_unpack = x
        pos_unpack = pos_unpack[:, :3].view(-1, 1, 3)
        x_per1, pos_per1 = self.per1(x_unpack, pos_unpack)
        row, col = edge_index

        ligand_ev = torch.squeeze(pos_unpack[edge_index[0]] - pos_unpack[edge_index[1]])
        ligand_ed = torch.squeeze(torch.norm(ligand_ev, dim=-1, p=2))

        edge_sca_feat = torch.cat([self.distance_expansion(ligand_ed), edge_attr], dim=-1).to(self.device)
        edge_vec_feat = self.vector_expansion(ligand_ev).to(self.device)

        msg_j_sca, msg_j_vec = self.msg1(node_features_sca=x_per1, node_features_vec=pos_per1,
                                         edge_features_sca=edge_sca_feat, edge_features_vec=edge_vec_feat,
                                         edge_index_node=col,
                                         dist_ij=ligand_ed,
                                         annealing=True)

        msg_j_sca, msg_j_vec = F.elu_(msg_j_sca), F.elu_(msg_j_vec)
        msg_j_sca = F.dropout(msg_j_sca, p=self.dropout, training=self.training)
        msg_j_vec = F.dropout(msg_j_vec, p=self.dropout, training=self.training)

        # Aggregate messages
        aggr_msg_sca = scatter_sum(msg_j_sca, row, dim=0, dim_size=x_per1.size(0))    # .view(N, -1) # (N, heads*H_per_head)
        aggr_msg_vec = scatter_sum(msg_j_vec, row, dim=0, dim_size=pos_per1.size(0))  # .view(N, -1, 3) # (N, heads*H_per_head, 3)

        x_out_sca, x_out_vec = self.centroid_lin(x_per1, pos_per1)
        out_sca = x_out_sca + aggr_msg_sca
        out_vec = x_out_vec + aggr_msg_vec

        out_sca = self.layernorm_sca(out_sca)
        out_vec = self.layernorm_vec(out_vec)
        out = self.out_transform(self.act_sca(out_sca), self.act_vec(out_vec))

        return out


class GATE_GRUConv_InterMol(nn.Module):
    """
    A layer gathering a GATEConv and a GRU layer
    First step in atomic embedding for intermolecular network
    """
    def __init__(
        self,
        in_channels: Tuple[int],
        out_channels: int,
        dropout: float,
        edge_dim: int,
        dim_hid: int = 1,
        cutoff: float = 10.,
        device: str = 'cuda'
    ) -> None:
        """
        Construct a GATE_GRUConv layer

        Args:
            in_channels (int): The input channel size
            out_channels (int): The output channels size
            dropout (float): The dropout rate
            edge_dim (int): The edge dimension
        """
        super().__init__()
        self.dropout = dropout
        self.dim_hid = dim_hid
        self.in_channels_src, self.in_channels_dst = in_channels

        self.per1_src = GVPerceptronVN(in_scalar=self.in_channels_src, in_vector=self.dim_hid,
                                       out_scalar=out_channels, out_vector=self.dim_hid,
                                       device=device)
        self.per1_dst = GVPerceptronVN(in_scalar=self.in_channels_dst, in_vector=self.dim_hid,
                                       out_scalar=out_channels, out_vector=self.dim_hid,
                                       device=device)
        self.msg1 = MessageModule(node_sca=out_channels, node_vec=dim_hid,
                                  edge_sca=2*edge_dim, edge_vec=dim_hid,
                                  out_sca=out_channels, out_vec=dim_hid,
                                  cutoff=cutoff, device=device)
        self.gru = GRUCell(out_channels, out_channels, device=device)
        self.centroid_lin = GVLinear(out_channels, dim_hid, out_channels, dim_hid, device=device)
        self.out_transform = GVLinear(out_channels, dim_hid, out_channels, dim_hid, device=device)
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_dim, device=device)
        self.vector_expansion = EdgeExpansion(dim_hid, device=device)
        self.act_sca = LeakyReLU()
        self.act_vec = VNLeakyReLU(1, device=device)
        self.layernorm_sca = LayerNorm([out_channels], device=device)
        self.layernorm_vec = LayerNorm([dim_hid, 3], device=device)

        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        """ GATEConv weights and biases are already initialised """
        glorot(self.per1_src.lin_vector_weight)
        glorot(self.per1_src.lin_vector2_weight)
        glorot(self.per1_src.scalar_to_vector_gates_weight)
        glorot(self.per1_src.lin_scalar_weight)

        zeros(self.per1_src.lin_vector_bias)
        zeros(self.per1_src.lin_vector2_bias)
        zeros(self.per1_src.scalar_to_vector_gates_bias)
        zeros(self.per1_src.lin_scalar_bias)

        glorot(self.per1_dst.lin_vector_weight)
        glorot(self.per1_dst.lin_vector2_weight)
        glorot(self.per1_dst.scalar_to_vector_gates_weight)
        glorot(self.per1_dst.lin_scalar_weight)

        zeros(self.per1_dst.lin_vector_bias)
        zeros(self.per1_dst.lin_vector2_bias)
        zeros(self.per1_dst.scalar_to_vector_gates_bias)
        zeros(self.per1_dst.lin_scalar_bias)

        glorot(self.centroid_lin.lin_vector_weight)
        glorot(self.centroid_lin.lin_vector2_weight)
        glorot(self.centroid_lin.scalar_to_vector_gates_weight)
        glorot(self.centroid_lin.lin_scalar_weight)

        zeros(self.centroid_lin.lin_vector_bias)
        zeros(self.centroid_lin.lin_vector2_bias)
        zeros(self.centroid_lin.scalar_to_vector_gates_bias)
        zeros(self.centroid_lin.lin_scalar_bias)

        glorot(self.out_transform.lin_vector_weight)
        glorot(self.out_transform.lin_vector2_weight)
        glorot(self.out_transform.scalar_to_vector_gates_weight)
        glorot(self.out_transform.lin_scalar_weight)

        zeros(self.out_transform.lin_vector_bias)
        zeros(self.out_transform.lin_vector2_bias)
        zeros(self.out_transform.scalar_to_vector_gates_bias)
        zeros(self.out_transform.lin_scalar_bias)

    def forward(
        self,
        x: Tuple[Tensor],
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process data graph through the layer

        Args:
            x (Tuple[Tensor]): Nodes features
            edge_index (Tensor): Edge indices
            edge_attr (Tensor): Edge attributes

        Output:
            Tensor: New node attributes
        """
        src_unpack, dst_unpack = x    # x[0] is src node features, x[1] is dst node coords
        x_src, pos_src = src_unpack
        pos_src = pos_src[:, :3].view(-1, 1, 3)
        x_dst, pos_dst = dst_unpack
        pos_dst = pos_dst[:, :3].view(-1, 1, 3)

        x_src_per1, pos_src_per1 = self.per1_src(x_src, pos_src)
        x_dst_per1, pos_dst_per1 = self.per1_dst(x_dst, pos_dst)

        row, col = edge_index
        ligand_ev = torch.squeeze(pos_src[edge_index[0]] - pos_dst[edge_index[1]])
        ligand_ed = torch.squeeze(torch.norm(ligand_ev, dim=-1, p=2))

        edge_sca_feat = torch.cat([self.distance_expansion(ligand_ed), edge_attr], dim=-1).to(self.device)
        edge_vec_feat = self.vector_expansion(ligand_ev).to(self.device)

        msg_j_src_sca, msg_j_src_vec = self.msg1(node_features_sca=x_dst_per1, node_features_vec=pos_dst_per1,
                                                 edge_features_sca=edge_sca_feat, edge_features_vec=edge_vec_feat,
                                                 edge_index_node=col,
                                                 dist_ij=ligand_ed,
                                                 annealing=True)
        msg_j_dst_sca, msg_j_dst_vec = self.msg1(node_features_sca=x_src_per1, node_features_vec=pos_src_per1,
                                                 edge_features_sca=edge_sca_feat, edge_features_vec=edge_vec_feat,
                                                 edge_index_node=row,
                                                 dist_ij=ligand_ed,
                                                 annealing=True)

        msg_j_sca, msg_j_vec = (msg_j_src_sca+msg_j_dst_sca)/2, (msg_j_src_vec+msg_j_dst_vec)/2
        msg_j_sca, msg_j_vec = F.elu_(msg_j_sca), F.elu_(msg_j_vec)
        msg_j_sca = F.dropout(msg_j_sca, p=self.dropout, training=self.training)
        msg_j_vec = F.dropout(msg_j_vec, p=self.dropout, training=self.training)

        # Aggregate messages
        aggr_msg_sca = scatter_sum(msg_j_sca, row, dim=0, dim_size=x_src_per1.size(0))    # .view(N, -1) # (N, heads*H_per_head)
        aggr_msg_vec = scatter_sum(msg_j_vec, row, dim=0, dim_size=pos_src_per1.size(0))  # .view(N, -1, 3) # (N, heads*H_per_head, 3)

        x_out_sca, x_out_vec = self.centroid_lin(x_src_per1, pos_src_per1)
        out_sca = x_out_sca + aggr_msg_sca
        out_vec = x_out_vec + aggr_msg_vec

        out_sca = self.layernorm_sca(out_sca)
        out_vec = self.layernorm_vec(out_vec)
        out_sca, out_vec = self.out_transform(self.act_sca(out_sca), self.act_vec(out_vec))

        return out_sca, out_vec


class GATGRUConv_IntraMol(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_gat: int,
        out_channels_gru: int,
        dropout: float,
        edge_dim: int,
        device: str = 'cuda',
        heads: Optional[int] = 1,
        add_self_loops: Optional[bool] = False,
    ) -> None:
        """
        Construct a GATGRUConv layer

        Args:
            in_channels (int): Input channels size
            out_channels_gat (int): GATConv output channels size (GRU input size)
            out_channels_gru (int): Output channel size
            dropout (float): Dropout rate
            edge_dim (int): Edge dimension
            heads (Optional[int]): Number of heads for the GATConv part. Default to 1
            add_self_loops (Optional[bool]): Add self loops. Default to False
        """
        super().__init__()
        self.dropout = dropout
        self.device = device

        self.per1 = GVPerceptronVN(in_channels, out_channels_gat, device=device)
        self.gat_conv = GATv2Conv(out_channels_gat,
                                  out_channels_gat,
                                  dropout=dropout,
                                  edge_dim=edge_dim,
                                  add_self_loops=add_self_loops,
                                  heads=heads)
        self.gru = GRUCell(out_channels_gat*heads, out_channels_gru)
        self.reset_parameters()

    def reset_parameters(self):
        """ GATConv weights and biases are already initialised """
        glorot(self.per1.lin_vector_weight)
        glorot(self.per1.lin_vector2_weight)
        glorot(self.per1.scalar_to_vector_gates_weight)
        glorot(self.per1.lin_scalar_weight)

        zeros(self.per1.lin_vector_bias)
        zeros(self.per1.lin_vector2_bias)
        zeros(self.per1.scalar_to_vector_gates_bias)
        zeros(self.per1.lin_scalar_bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x_unpack, pos_unpack = x
        pos_unpack = pos_unpack[:, :3].view(-1, 1, 3)
        x_per1, pos_per1 = self.per1(x_unpack, pos_unpack)
        row, col = edge_index

        ligand_ev = torch.squeeze(pos_unpack[edge_index[0]] - pos_unpack[edge_index[1]])
        ligand_ed = torch.squeeze(torch.norm(ligand_ev, dim=-1, p=2))

        return x


def glorot(value: Any):
    """ From pyg.nn.inits """
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)


def zeros(value:Any):
    """ From pyg.nn.inits """
    constant(value, 0.)


def constant(value:Any, fill_value:float):
    """ From pyg.nn.inits """
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)
