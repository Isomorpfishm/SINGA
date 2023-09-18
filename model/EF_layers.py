"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import copy
import math
from typing import Dict, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
import torch_geometric as pyg

from e3nn import o3
from e3nn.o3 import FromS2Grid, ToS2Grid


class EdgeDegreeEmbedding(nn.Module):
    """
    Args:
        sphere_channels (int):      Number of spherical channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_rotation (list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated

        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features

        rescale_factor (float):     Rescale the sum aggregation
    """
    def __init__(
        self,
        sphere_channels,
        lmax_list,
        mmax_list,
        SO3_rotation,
        mappingReduced,
        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding,
        rescale_factor,
        device: str = 'cuda',
    ):
        super(EdgeDegreeEmbedding, self).__init__()
        self.sphere_channels = sphere_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)
        self.SO3_rotation = SO3_rotation
        self.mappingReduced = mappingReduced

        self.m_0_num_coefficients = self.mappingReduced.m_size[0]
        self.m_all_num_coefficients = len(self.mappingReduced.l_harmonic)

        # Create edge scalar (invariant to rotation) features
        # Embedding function of the atomic numbers
        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_atom_edge_embedding = use_atom_edge_embedding

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1], device=device)
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1], device=device)
            nn.init.uniform(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None

        # Embedding function of distance
        self.edge_channels_list.append(self.m_0_num_coefficients * self.sphere_channels)
        self.rad_func = RadialFunction(self.edge_channels_list, device=device)
        self.rescale_factor = rescale_factor
        self.device = device

    def forward(
        self,
        atomic_numbers: Union[Tensor, Dict],
        edge_distance: Tensor,
        edge_index: Tensor,
        hetero: bool,
    ):
        assert hetero is not None, "Please specify args: hetero"
        if self.use_atom_edge_embedding:
            if hetero:
                assert list(atomic_numbers.keys()) == ['source', 'target'], "Error: atomic_numbers incorrect for Hetero case"
                source_atomic_num, target_atomic_num = atomic_numbers['source'], atomic_numbers['target']
                source_element = source_atomic_num[edge_index[0]]
                target_element = target_atomic_num[edge_index[1]]
            else:
                source_element = atomic_numbers[edge_index[0]]
                target_element = atomic_numbers[edge_index[1]]
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
        else:
            x_edge = edge_distance

        x_edge_m_0 = self.rad_func(x_edge)
        x_edge_m_0 = x_edge_m_0.reshape(-1, self.m_0_num_coefficients, self.sphere_channels)
        x_edge_m_pad = torch.zeros((
            x_edge_m_0.shape[0],
            (self.m_all_num_coefficients - self.m_0_num_coefficients),
            self.sphere_channels),
            device=self.device,
        )
        x_edge_m_all = torch.cat((x_edge_m_0, x_edge_m_pad), dim=1)
        
        x_edge_embedding = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.sphere_channels,
            device=self.device,
            dtype=x_edge_m_all.dtype,
        )
        x_edge_embedding.set_embedding(x_edge_m_all)
        x_edge_embedding.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # Reshape the spherical harmonics based on l (degree)
        x_edge_embedding._l_primary(self.mappingReduced)

        # Rotate back the irreps
        x_edge_embedding._rotate_inv(self.SO3_rotation, self.mappingReduced)

        # Compute the sum of the incoming neighbouring messages for each target node
        x_edge_embedding._reduce_edge(edge_index[1], atomic_numbers.shape[0])
        x_edge_embedding.embedding = x_edge_embedding.embedding / self.rescale_factor

        return x_edge_embedding


class SO2_m_Convolution(nn.Module):
    """
    SO(2) Conv: Perform an SO(2) convolution on features corresponding to +- m

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
    """
    def __init__(
        self,
        m,
        sphere_channels,
        m_output_channels,
        lmax_list,
        mmax_list,
        device: str = 'cuda',
    ):
        super(SO2_m_Convolution, self).__init__()
        self.m = m
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)
        self.device = device

        num_channels = 0
        for i in range(self.num_resolutions):
            num_coefficents = 0
            if self.mmax_list[i] >= self.m:
                num_coefficents = self.lmax_list[i] - self.m + 1
            num_channels = num_channels + num_coefficents * self.sphere_channels
        assert num_channels > 0

        self.fc = Linear(num_channels,
                         2 * self.m_output_channels * (num_channels // self.sphere_channels),
                         bias=False,
                         device=device)
        self.fc.weight.data.mul_(1 / math.sqrt(2))

    def forward(self, x_m):
        x_m = self.fc(x_m)
        x_r = x_m.narrow(2, 0, self.fc.out_features // 2)
        x_i = x_m.narrow(2, self.fc.out_features // 2, self.fc.out_features // 2)
        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1)  # x_r[:, 0] - x_i[:, 1]
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1)  # x_r[:, 1] + x_i[:, 0]
        x_out = torch.cat((x_m_r, x_m_i), dim=1)

        return x_out


class SO2_Convolution(nn.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders)

    Args:
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        mappingReduced (CoefficientMappingModule): Used to extract a subset of m components
        internal_weights (bool):    If True, not using radial function to multiply inputs features
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
        extra_m0_output_channels (int): If not None, return `out_embedding` (SO3_Embedding) and `extra_m0_features` (Tensor).
    """
    def __init__(
        self,
        sphere_channels: int,
        m_output_channels: int,
        lmax_list: list,
        mmax_list: list,
        mappingReduced,
        edge_channels_list = None,
        extra_m0_output_channels = None,
        internal_weights: bool = True,
        device: str = 'cuda',
    ):
        super(SO2_Convolution, self).__init__()
        self.device = device
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.mappingReduced = mappingReduced
        self.num_resolutions = len(lmax_list)
        self.internal_weights = internal_weights
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.extra_m0_output_channels = extra_m0_output_channels

        num_channels_rad = 0  # for radial function
        num_channels_m0 = 0

        for i in range(self.num_resolutions):
            num_coefficients = self.lmax_list[i] + 1
            num_channels_m0 = num_channels_m0 + num_coefficients * self.sphere_channels

        # SO(2) convolution for m = 0
        m0_output_channels = self.m_output_channels * (num_channels_m0 // self.sphere_channels)
        if self.extra_m0_output_channels is not None:
            m0_output_channels = m0_output_channels + self.extra_m0_output_channels
        self.fc_m0 = Linear(num_channels_m0, m0_output_channels, device=device)
        num_channels_rad = num_channels_rad + self.fc_m0.in_features

        # SO(2) convolution for non-zero m
        self.so2_m_conv = nn.ModuleList()
        for m in range(1, max(self.mmax_list) + 1):
            self.so2_m_conv.append(
                SO2_m_Convolution(
                    m,
                    self.sphere_channels,
                    self.m_output_channels,
                    self.lmax_list,
                    self.mmax_list,
                    device=self.device,
                )
            )
            num_channels_rad = num_channels_rad + self.so2_m_conv[-1].fc.in_features

        # Embedding function of distance
        self.rad_func = None
        if not self.internal_weights:
            assert self.edge_channels_list is not None
            self.edge_channels_list.append(int(num_channels_rad))
            self.rad_func = RadialFunction(self.edge_channels_list, device=device)

    def forward(self, x, x_edge):
        num_edges = len(x_edge)
        out = []

        # Reshape the spherical harmonics based on m (order)
        x._m_primary(self.mappingReduced)

        # radial function
        if self.rad_func is not None:
            x_edge = self.rad_func(x_edge)
        offset_rad = 0

        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        x_0 = x.embedding.narrow(1, 0, self.mappingReduced.m_size[0])
        x_0 = x_0.reshape(num_edges, -1)
        if self.rad_func is not None:
            x_edge_0 = x_edge.narrow(1, 0, self.fc_m0.in_features)
            x_0 = x_0 * x_edge_0
        x_0 = self.fc_m0(x_0)

        x_0_extra = None
        # extract extra m0 features
        if self.extra_m0_output_channels is not None:
            x_0_extra = x_0.narrow(-1, 0, self.extra_m0_output_channels)
            x_0 = x_0.narrow(-1, self.extra_m0_output_channels,
                             (self.fc_m0.out_features - self.extra_m0_output_channels))

        x_0 = x_0.view(num_edges, -1, self.m_output_channels)
        # x.embedding[:, 0 : self.mappingReduced.m_size[0]] = x_0
        out.append(x_0)
        offset_rad = offset_rad + self.fc_m0.in_features

        # Compute the values for the m > 0 coefficients
        offset = self.mappingReduced.m_size[0]
        for m in range(1, max(self.mmax_list) + 1):
            # Get the m order coefficients
            x_m = x.embedding.narrow(1, offset, 2 * self.mappingReduced.m_size[m])
            x_m = x_m.reshape(num_edges, 2, -1)

            # Perform SO(2) convolution
            if self.rad_func is not None:
                x_edge_m = x_edge.narrow(1, offset_rad, self.so2_m_conv[m - 1].fc.in_features)
                x_edge_m = x_edge_m.reshape(num_edges, 1, self.so2_m_conv[m - 1].fc.in_features)
                x_m = x_m * x_edge_m
            x_m = self.so2_m_conv[m - 1](x_m)
            x_m = x_m.view(num_edges, -1, self.m_output_channels)
            # x.embedding[:, offset : offset + 2 * self.mappingReduced.m_size[m]] = x_m
            out.append(x_m)
            offset = offset + 2 * self.mappingReduced.m_size[m]
            offset_rad = offset_rad + self.so2_m_conv[m - 1].fc.in_features

        out = torch.cat(out, dim=1)
        out_embedding = SO3_Embedding(
            0,
            x.lmax_list.copy(),
            self.m_output_channels,
            device=self.device,
            dtype=x.dtype
        )
        out_embedding.set_embedding(out)
        out_embedding.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # Reshape the spherical harmonics based on l (degree)
        out_embedding._l_primary(self.mappingReduced)

        if self.extra_m0_output_channels is not None:
            return out_embedding, x_0_extra
        else:
            return out_embedding


class SO2EquivariantGraphAttention(nn.Module):
    """
    SO2EquivariantGraphAttention: Perform MLP attention + non-linear message passing
        SO(2) Convolution with radial function -> S2 Activation -> SO(2) Convolution -> attention weights and non-linear messages
        attention weights * non-linear messages -> Linear

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        output_channels (int):      Number of output channels
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_rotation (list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features
        use_m_share_rad (bool):     Whether all m components within a type-L vector of one channel share radial function weights

        activation (str):           Type of activation function
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
    """
    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        num_heads: int,
        attn_alpha_channels,
        attn_value_channels,
        output_channels,

        lmax_list: list,
        mmax_list: list,

        SO3_rotation,
        mappingReduced,
        SO3_grid,
        max_num_elements,
        edge_channels_list: list,

        use_atom_edge_embedding: bool = True,
        use_m_share_rad: bool = False,
        activation: str = 'scaled_silu',
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        use_gate_act: bool = False,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.0,
        device: str = 'cuda',
    ):
        super(SO2EquivariantGraphAttention, self).__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)
        self.device = device

        self.SO3_rotation = SO3_rotation
        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1], device=device)
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1], device=device)
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None

        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.use_gate_act = use_gate_act
        self.use_sep_s2_act = use_sep_s2_act

        assert not self.use_s2_act_attn  # since this is not used

        # Create SO(2) convolution blocks
        extra_m0_output_channels = None
        if not self.use_s2_act_attn:
            extra_m0_output_channels = self.num_heads * self.attn_alpha_channels
            if self.use_gate_act:
                extra_m0_output_channels = extra_m0_output_channels + max(self.lmax_list) * self.hidden_channels
            else:
                if self.use_sep_s2_act:
                    extra_m0_output_channels = extra_m0_output_channels + self.hidden_channels

        if self.use_m_share_rad:
            self.edge_channels_list = self.edge_channels_list + [2 * self.sphere_channels * (max(self.lmax_list) + 1)]
            self.rad_func = RadialFunction(self.edge_channels_list, device=device)
            expand_index = torch.zeros([(max(self.lmax_list) + 1) ** 2], device=device).long()
            for l in range(max(self.lmax_list) + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                expand_index[start_idx: (start_idx + length)] = l
            self.register_buffer('expand_index', expand_index)

        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=(
                False if not self.use_m_share_rad
                else True
            ),
            edge_channels_list=(
                self.edge_channels_list if not self.use_m_share_rad
                else None
            ),
            extra_m0_output_channels=extra_m0_output_channels,  # for attention weights and/or gate activation
            device=self.device,
        )

        if self.use_s2_act_attn:
            self.alpha_norm = None
            self.alpha_act = None
            self.alpha_dot = None
        else:
            if self.use_attn_renorm:
                self.alpha_norm = nn.LayerNorm(self.attn_alpha_channels, device=self.device)
            else:
                self.alpha_norm = nn.Identity()
            self.alpha_act = SmoothLeakyReLU()
            self.alpha_dot = nn.Parameter(torch.randn(self.num_heads, self.attn_alpha_channels))
            # pyg.nn.inits.glorot(self.alpha_dot) # Following GATv2
            std = 1.0 / math.sqrt(self.attn_alpha_channels)
            nn.init.uniform_(self.alpha_dot, -std, std)

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = nn.Dropout(alpha_drop)

        if self.use_gate_act:
            self.gate_act = GateActivation(
                lmax=max(self.lmax_list),
                mmax=max(self.mmax_list),
                num_channels=self.hidden_channels,
                device=self.device,
            )
        else:
            if self.use_sep_s2_act:
                # separable S2 activation
                self.s2_act = SeparableS2Activation(
                    lmax=max(self.lmax_list),
                    mmax=max(self.mmax_list),
                )
            else:
                # S2 activation
                self.s2_act = S2Activation(
                    lmax=max(self.lmax_list),
                    mmax=max(self.mmax_list)
                )

        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.num_heads * self.attn_value_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=(
                self.num_heads if self.use_s2_act_attn
                else None
            ),  # for attention weights
            device=self.device,
        )

        self.proj = SO3_LinearV2(self.num_heads * self.attn_value_channels,
                                 self.output_channels,
                                 lmax=self.lmax_list[0],
                                 device=self.device,
                                 )

    def forward(
        self,
        x,
        atomic_numbers,
        edge_distance,
        edge_index,
    ):
        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        if self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
        else:
            x_edge = edge_distance

        x_source = x.clone()
        x_target = x.clone()
        x_source._expand_edge(edge_index[0, :])
        x_target._expand_edge(edge_index[1, :])

        x_message_data = torch.cat((x_source.embedding, x_target.embedding), dim=2)
        x_message = SO3_Embedding(
            0,
            x_target.lmax_list.copy(),
            x_target.num_channels * 2,
            device=self.device,
            dtype=x_target.dtype,
        )
        x_message.set_embedding(x_message_data)
        x_message.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # radial function (scale all m components within a type-L vector of one channel with the same weight)
        if self.use_m_share_rad:
            x_edge_weight = self.rad_func(x_edge)
            x_edge_weight = x_edge_weight.reshape(-1, (max(self.lmax_list) + 1), 2 * self.sphere_channels)
            x_edge_weight = torch.index_select(x_edge_weight, dim=1,
                                               index=self.expand_index)  # [E, (L_max + 1) ** 2, C]
            x_message.embedding = x_message.embedding * x_edge_weight

        # Rotate the irreps to align with the edge
        x_message._rotate(self.SO3_rotation, self.lmax_list, self.mmax_list)

        # First SO(2)-convolution
        if self.use_s2_act_attn:
            x_message = self.so2_conv_1(x_message, x_edge)
        else:
            x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)

        # Activation
        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels
        if self.use_gate_act:
            # Gate activation
            x_0_gating = x_0_extra.narrow(1, x_alpha_num_channels,
                                          x_0_extra.shape[1] - x_alpha_num_channels)  # for activation
            x_0_alpha = x_0_extra.narrow(1, 0, x_alpha_num_channels)  # for attention weights
            x_message.embedding = self.gate_act(x_0_gating, x_message.embedding)
        else:
            if self.use_sep_s2_act:
                x_0_gating = x_0_extra.narrow(1, x_alpha_num_channels,
                                              x_0_extra.shape[1] - x_alpha_num_channels)  # for activation
                x_0_alpha = x_0_extra.narrow(1, 0, x_alpha_num_channels)  # for attention weights
                x_message.embedding = self.s2_act(x_0_gating, x_message.embedding, self.SO3_grid)
            else:
                x_0_alpha = x_0_extra
                x_message.embedding = self.s2_act(x_message.embedding, self.SO3_grid)

        # Second SO(2)-convolution
        if self.use_s2_act_attn:
            x_message, x_0_extra = self.so2_conv_2(x_message, x_edge)
        else:
            x_message = self.so2_conv_2(x_message, x_edge)

        # Attention weights
        if self.use_s2_act_attn:
            alpha = x_0_extra
        else:
            x_0_alpha = x_0_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels)
            x_0_alpha = self.alpha_norm(x_0_alpha)
            x_0_alpha = self.alpha_act(x_0_alpha)
            alpha = torch.einsum('bik, ik -> bi', x_0_alpha, self.alpha_dot)

        alpha = pyg.utils.softmax(alpha, edge_index[1])
        alpha = alpha.reshape(alpha.shape[0], 1, self.num_heads, 1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)

        # Attention weights * non-linear messages
        attn = x_message.embedding
        attn = attn.reshape(attn.shape[0], attn.shape[1], self.num_heads, self.attn_value_channels)
        attn = attn * alpha
        attn = attn.reshape(attn.shape[0], attn.shape[1], self.num_heads * self.attn_value_channels)
        x_message.embedding = attn

        # Rotate back the irreps
        x_message._rotate_inv(self.SO3_rotation, self.mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        x_message._reduce_edge(edge_index[1], len(x.embedding))

        # Project
        out_embedding = self.proj(x_message)

        return out_embedding


class FeedForwardNetwork(nn.Module):
    """
    FeedForwardNetwork: Perform feedforward network with S2 activation or gate activation

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        activation (str):           Type of activation function
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs.
        use_sep_s2_act (bool):      If `True`, use separable grid MLP when `use_grid_mlp` is True.
    """
    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        output_channels,
        lmax_list: list,
        mmax_list: list,
        SO3_grid,

        activation: bool = 'scaled_silu',
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        device: str = 'cuda',
    ):
        super(FeedForwardNetwork, self).__init__()
        self.device = device
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        self.SO3_grid = SO3_grid
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        self.max_lmax = max(self.lmax_list)

        self.so3_linear_1 = SO3_LinearV2(self.sphere_channels_all, self.hidden_channels, lmax=self.max_lmax, device=device)
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                self.scalar_mlp = nn.Sequential(
                    Linear(self.sphere_channels_all, self.hidden_channels, bias=True, device=device),
                    nn.SiLU(),
                )
            else:
                self.scalar_mlp = None
            self.grid_mlp = nn.Sequential(
                Linear(self.hidden_channels, self.hidden_channels, bias=False, device=device),
                nn.SiLU(),
                Linear(self.hidden_channels, self.hidden_channels, bias=False, device=device),
                nn.SiLU(),
                Linear(self.hidden_channels, self.hidden_channels, bias=False, device=device)
            )
        else:
            if self.use_gate_act:
                self.gating_linear = Linear(self.sphere_channels_all,
                                            self.max_lmax * self.hidden_channels,
                                            device=device)
                self.gate_act = GateActivation(self.max_lmax, self.max_lmax, self.hidden_channels, device=self.device)
            else:
                if self.use_sep_s2_act:
                    self.gating_linear = Linear(self.sphere_channels_all,
                                                self.hidden_channels,
                                                device=device)
                    self.s2_act = SeparableS2Activation(self.max_lmax, self.max_lmax)
                else:
                    self.gating_linear = None
                    self.s2_act = S2Activation(self.max_lmax, self.max_lmax)
        self.so3_linear_2 = SO3_LinearV2(self.hidden_channels, self.output_channels, lmax=self.max_lmax, device=self.device)

    def forward(self, input_embedding):
        gating_scalars = None

        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                gating_scalars = self.scalar_mlp(input_embedding.embedding.narrow(1, 0, 1))
        else:
            if self.gating_linear is not None:
                gating_scalars = self.gating_linear(input_embedding.embedding.narrow(1, 0, 1))

        input_embedding = self.so3_linear_1(input_embedding)

        if self.use_grid_mlp:
            # Project to grid
            input_embedding_grid = input_embedding.to_grid(self.SO3_grid, lmax=self.max_lmax)
            # Perform point-wise operations
            input_embedding_grid = self.grid_mlp(input_embedding_grid)
            # Project back to spherical harmonic coefficients
            input_embedding._from_grid(input_embedding_grid, self.SO3_grid, lmax=self.max_lmax)

            if self.use_sep_s2_act:
                input_embedding.embedding = torch.cat(
                    (gating_scalars, input_embedding.embedding.narrow(1, 1, input_embedding.embedding.shape[1]-1)),
                    dim=1
                )
        else:
            if self.use_gate_act:
                input_embedding.embedding = self.gate_act(gating_scalars, input_embedding.embedding)
            else:
                if self.use_sep_s2_act:
                    input_embedding.embedding = self.s2_act(gating_scalars, input_embedding.embedding, self.SO3_grid)
                else:
                    input_embedding.embedding = self.s2_act(input_embedding.embedding, self.SO3_grid)

        input_embedding = self.so3_linear_2(input_embedding)

        return input_embedding


class TransBlockV2(nn.Module):
    """
    Args:
        sphere_channels (int):      Number of spherical channels
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_rotation (list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features
        use_m_share_rad (bool):     Whether all m components within a type-L vector of one channel share radial function weights

        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFN.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh'])

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN
    """
    def __init__(
        self,
        sphere_channels: int,
        attn_hidden_channels: int,
        attn_alpha_channels: int,
        attn_value_channels: int,
        ffn_hidden_channels: int,
        output_channels: int,

        edge_channels_list: list,
        lmax_list: list,
        mmax_list: list,

        SO3_rotation,
        mappingReduced,
        SO3_grid,

        num_heads: int,
        max_num_elements: int,

        use_atom_edge_embedding: bool = True,
        use_m_share_rad: bool = False,
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,

        attn_activation: str = 'silu',
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = 'silu',

        norm_type: str = 'rms_norm_sh',
        alpha_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        proj_drop: float = 0.0,
        device: str = 'cuda',
    ):
        super(TransBlockV2, self).__init__()
        self.device = device
        max_lmax = max(lmax_list)
        self.norm_1 = get_normalization_layer(norm_type, lmax=max_lmax, num_channels=sphere_channels, device=device)
        self.norm_2 = get_normalization_layer(norm_type, lmax=max_lmax, num_channels=sphere_channels, device=device)

        self.ga = SO2EquivariantGraphAttention(
            sphere_channels=sphere_channels,
            hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            output_channels=sphere_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_rotation=SO3_rotation,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            max_num_elements=max_num_elements,
            edge_channels_list=edge_channels_list,
            use_atom_edge_embedding=use_atom_edge_embedding,
            use_m_share_rad=use_m_share_rad,
            activation=attn_activation,
            use_s2_act_attn=use_s2_act_attn,
            use_attn_renorm=use_attn_renorm,
            use_gate_act=use_gate_act,
            use_sep_s2_act=use_sep_s2_act,
            alpha_drop=alpha_drop,
            device=device,
        )

        self.drop_path = GraphDropPath(drop_path_rate, device=self.device) if drop_path_rate > 0. else None
        self.proj_drop = EquivariantDropoutArraySphericalHarmonics(proj_drop,
                                                                   drop_graph=False,
                                                                   device=self.device) if proj_drop > 0.0 else None

        self.ffn = FeedForwardNetwork(
            sphere_channels=sphere_channels,
            hidden_channels=ffn_hidden_channels,
            output_channels=output_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_grid=SO3_grid,
            activation=ffn_activation,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act,
            device=device,
        )

        if sphere_channels != output_channels:
            self.ffn_shortcut = SO3_LinearV2(sphere_channels, output_channels, lmax=max_lmax, device=self.device)
        else:
            self.ffn_shortcut = None

    def forward(
        self,
        x,  # SO3_Embedding
        atomic_numbers,
        edge_distance,
        edge_index,
        batch  # for GraphDropPath
    ):
        output_embedding = x
        x_res = output_embedding.embedding
        output_embedding.embedding = self.norm_1(output_embedding.embedding)
        output_embedding = self.ga(output_embedding,
                                   atomic_numbers,
                                   edge_distance,
                                   edge_index)

        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(output_embedding.embedding, batch)
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(output_embedding.embedding, batch)

        output_embedding.embedding = output_embedding.embedding + x_res

        x_res = output_embedding.embedding
        output_embedding.embedding = self.norm_2(output_embedding.embedding)
        output_embedding = self.ffn(output_embedding)

        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(output_embedding.embedding, batch)
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(output_embedding.embedding, batch)

        if self.ffn_shortcut is not None:
            shortcut_embedding = SO3_Embedding(
                0,
                output_embedding.lmax_list.copy(),
                self.ffn_shortcut.in_features,
                device=self.device,
                dtype=output_embedding.dtype
            )
            shortcut_embedding.set_embedding(x_res)
            shortcut_embedding.set_lmax_mmax(output_embedding.lmax_list.copy(), output_embedding.lmax_list.copy())
            shortcut_embedding = self.ffn_shortcut(shortcut_embedding)
            x_res = shortcut_embedding.embedding

        output_embedding.embedding = output_embedding.embedding + x_res

        return output_embedding


class SO3_Embedding():
    def __init__(
        self,
        length: int,
        lmax_list: list,
        num_channels: int,
        dtype = torch.long,
        device: str = 'cuda',
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.device = device
        self.dtype = dtype
        self.num_resolutions = len(lmax_list)
        self.num_coefficients = 0

        for i in range(self.num_resolutions):
            self.num_coefficients = self.num_coefficients + int((lmax_list[i]+1)**2)

        embedding = torch.zeros(
            length,
            self.num_coefficients,
            self.num_channels,
            device=self.device,
            dtype=self.dtype,
        )

        self.set_embedding(embedding)
        self.set_lmax_mmax(lmax_list, lmax_list.copy())

    # Clone an embedding of irreps
    def clone(self):
        clone = SO3_Embedding(
            length=0,
            lmax_list=self.lmax_list.copy(),
            num_channels=self.num_channels,
            device=self.device,
            dtype=self.dtype,
        )
        clone.set_embedding(self.embedding.clone())
        return clone

    # Initialise an embedding of irreps
    def set_embedding(self, embedding) -> None:
        self.length = len(embedding)
        self.embedding = embedding

    # Set the maximum order to be the maximum degree
    def set_lmax_mmax(self, lmax_list, mmax_list) -> None:
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list

    # Expand the node embeddings to the number of edges
    def _expand_edge(self, edge_index):
        embedding = self.embedding[edge_index]
        self.set_embedding(embedding)

    def expand_edge(self, edge_index):
        x_expand = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.num_channels,
            self.device,
            self.dtype,
        )
        x_expand.set_embedding(self.embedding[edge_index])
        return x_expand

    # Compute the sum of the embeddings of the neighbourhood
    def _reduce_edge(self, edge_index, num_nodes):
        new_embedding = torch.zeros(
            num_nodes,
            self.num_coefficients,
            self.num_channels,
            device=self.device,
            dtype=self.embedding.dtype,
        )
        new_embedding.index_add_(0, edge_index, self.embedding)
        self.set_embedding(new_embedding)

    # Reshape the embedding 1 -> m
    def _m_primary(self, mapping):
        self.embedding = torch.einsum("nac, ba -> nbc", self.embedding, mapping.to_m)

    # Reshape the embedding m -> 1
    def _l_primary(self, mapping):
        self.embedding = torch.einsum("nac, ab -> nbc", self.embedding, mapping.to_m)

    # Rotate the embedding
    def _rotate(self, SO3_rotation, lmax_list, mmax_list):
        if self.num_resolutions == 1:
            embedding_rotate = SO3_rotation[0].rotate(self.embedding, lmax_list[0], mmax_list[0])
        else:
            offset = 0
            embedding_rotate = torch.tensor([], device=self.device, dtype=self.dtype)
            for i in range(self.num_resolutions):
                num_coefficients = int((self.lmax_list[i] + 1) ** 2)
                embedding_i = self.embedding[:, offset: offset + num_coefficients]
                embedding_rotate = torch.cat(
                    [embedding_rotate,
                            SO3_rotation[i].rotate(embedding_i, lmax_list[i], mmax_list[i])
                     ], dim=1)
                offset = offset + num_coefficients

        self.embedding = embedding_rotate
        self.set_lmax_mmax(lmax_list.copy(), mmax_list.copy())

    # Rotate the embedding by the inverse of the rotation matrix
    def _rotate_inv(self, SO3_rotation, mappingReduced):
        if self.num_resolutions == 1:
            embedding_rotate = SO3_rotation[0].rotate_inv(self.embedding, self.lmax_list[0], self.mmax_list[0])
        else:
            offset = 0
            embedding_rotate = torch.tensor([], device=self.device, dtype=self.dtype)
            for i in range(self.num_resolutions):
                num_coefficients = mappingReduced.res_size[i]
                embedding_i = self.embedding[:, offset: offset + num_coefficients]
                embedding_rotate = torch.cat([
                    embedding_rotate,
                    SO3_rotation[i].rotate_inv(embedding_i, self.lmax_list[i], self.mmax_list[i])], dim=1)
                offset = offset + num_coefficients
        self.embedding = embedding_rotate

        # Assume mmax = lmax when rotating back
        for i in range(self.num_resolutions):
            self.mmax_list[i] = int(self.lmax_list[i])
        self.set_lmax_mmax(self.lmax_list, self.mmax_list)

    # Compute point-wise spherical non-linearity
    def _grid_act(self, SO3_grid, act, mappingReduced):
        offset = 0
        for i in range(self.num_resolutions):
            num_coefficients = mappingReduced.res_size[i]
            if self.num_resolutions == 1:
                x_res = self.embedding
            else:
                x_res = self.embedding[:, offset: offset + num_coefficients].contiguous()
            to_grid_mat = SO3_grid[self.lmax_list[i]][self.mmax_list[i]].get_to_grid_mat(self.device)
            from_grid_mat = SO3_grid[self.lmax_list[i]][self.mmax_list[i]].get_from_grid_mat(self.device)

            x_grid = torch.einsum("bai, zic -> zbac", to_grid_mat, x_res)
            x_grid = act(x_grid)
            x_res = torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)
            if self.num_resolutions == 1:
                self.embedding = x_res
            else:
                self.embedding[:, offset: offset + num_coefficients] = x_res
            offset = offset + num_coefficients

    # Compute a sample of the grid
    def to_grid(self, SO3_grid, lmax=-1):
        if lmax == -1:
            lmax = max(self.lmax_list)

        to_grid_mat_lmax = SO3_grid[lmax][lmax].get_to_grid_mat(self.device)
        grid_mapping = SO3_grid[lmax][lmax].mapping
        offset = 0
        x_grid = torch.tensor([], device=self.device)

        for i in range(self.num_resolutions):
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)
            if self.num_resolutions == 1:
                x_res = self.embedding
            else:
                x_res = self.embedding[:, offset: offset + num_coefficients].contiguous()
            to_grid_mat = to_grid_mat_lmax[:, :, grid_mapping.coefficient_idx(self.lmax_list[i], self.lmax_list[i])]
            x_grid = torch.cat([x_grid, torch.einsum("bai, zic -> zbac", to_grid_mat, x_res)], dim=3)
            offset = offset + num_coefficients

        return x_grid

    # Compute irreps from grid representation
    def _from_grid(self, x_grid, SO3_grid, lmax=-1):
        if lmax == -1:
            lmax = max(self.lmax_list)

        from_grid_mat_lmax = SO3_grid[lmax][lmax].get_from_grid_mat(self.device)
        grid_mapping = SO3_grid[lmax][lmax].mapping
        offset = 0
        offset_channel = 0

        for i in range(self.num_resolutions):
            from_grid_mat = from_grid_mat_lmax[:, :, grid_mapping.coefficient_idx(self.lmax_list[i], self.lmax_list[i])]
            if self.num_resolutions == 1:
                temp = x_grid
            else:
                temp = x_grid[:, :, :, offset_channel: offset_channel + self.num_channels]
            x_res = torch.einsum("bai, zbac -> zic", from_grid_mat, temp)
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)

            if self.num_resolutions == 1:
                self.embedding = x_res
            else:
                self.embedding[:, offset: offset + num_coefficients] = x_res

            offset = offset + num_coefficients
            offset_channel = offset_channel + self.num_channels


class SO3_Rotation(nn.Module):
    """
    Helper functions for Wigner-D rotations

    Args:
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
    """
    def __init__(self, lmax, device: str = 'cuda'):
        super().__init__()
        self.lmax = lmax
        self.mapping = CoefficientMappingModule([self.lmax], [self.lmax], device=device)
        self.device = device

    def set_wigner(self, rot_mat3x3):
        self.dtype = rot_mat3x3.dtype
        length = len(rot_mat3x3)
        self.wigner = self.RotationToWignerDMatrix(rot_mat3x3, 0, self.lmax)
        self.wigner_inv = torch.transpose(self.wigner, 1, 2).contiguous()
        self.wigner = self.wigner.detach()
        self.wigner_inv = self.wigner_inv.detach()

    # Rotate the embedding
    def rotate(self, embedding, out_lmax, out_mmax):
        out_mask = self.mapping.coefficient_idx(out_lmax, out_mmax)
        wigner = self.wigner[:, out_mask, :]
        return torch.bmm(wigner, embedding)

    # Rotate the embedding by the inverse of the rotation matrix
    def rotate_inv(self, embedding, in_lmax, in_mmax):
        in_mask = self.mapping.coefficient_idx(in_lmax, in_mmax)
        wigner_inv = self.wigner_inv[:, :, in_mask]
        wigner_inv_rescale = self.mapping.get_rotate_inv_rescale(in_lmax, in_mmax)
        wigner_inv = wigner_inv * wigner_inv_rescale
        return torch.bmm(wigner_inv, embedding)

    # Compute Wigner matrices from rotation matrix
    def RotationToWignerDMatrix(self, edge_rot_mat, start_lmax, end_lmax):
        x = edge_rot_mat @ edge_rot_mat.new_tensor([0.0, 1.0, 0.0])
        alpha, beta = o3.xyz_to_angles(x)
        R = (
                o3.angles_to_matrix(
                    alpha, beta, torch.zeros_like(alpha)
                ).transpose(-1, -2)
                @ edge_rot_mat
        )
        gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])
        size = (end_lmax + 1) ** 2 - (start_lmax) ** 2
        wigner = torch.zeros(len(alpha), size, size, device=self.device)
        start = 0

        for lmax in range(start_lmax, end_lmax + 1):
            block = wigner_D(lmax, alpha, beta, gamma)
            end = start + block.size()[1]
            wigner[:, start:end, start:end] = block
            start = end

        return wigner.detach()


class SO3_Grid(nn.Module):
    """
    Helper functions for grid representation of the irreps

    Args:
        lmax (int):   Maximum degree of the spherical harmonics
        mmax (int):   Maximum order of the spherical harmonics
    """
    def __init__(
        self,
        lmax: int,
        mmax: int,
        normalization: str = 'integral',
        resolution = None,
        device: str = 'cuda',
    ):
        super().__init__()
        self.device = device
        self.lmax = lmax
        self.mmax = mmax
        self.lat_resolution = 2 * (self.lmax + 1)

        if lmax == mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * (self.mmax) + 1
        if resolution is not None:
            self.lat_resolution = resolution
            self.long_resolution = resolution

        self.mapping = CoefficientMappingModule([self.lmax], [self.lmax], device=self.device)
        to_grid = ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization=normalization, #normalization="integral",
            device=device,
        )

        to_grid_mat = torch.einsum("mbi, am -> bai", to_grid.shb, to_grid.sha).detach()
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l ** 2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                to_grid_mat[:, :, start_idx : (start_idx + length)] = to_grid_mat[:, :, start_idx : (start_idx + length)] * rescale_factor
        to_grid_mat = to_grid_mat[:, :, self.mapping.coefficient_idx(self.lmax, self.mmax)]

        from_grid = FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=normalization, #normalization="integral",
            device=device,
        )
        from_grid_mat = torch.einsum("am, mbi -> bai", from_grid.sha, from_grid.shb).detach()
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l ** 2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                from_grid_mat[:, :, start_idx : (start_idx + length)] = from_grid_mat[:, :, start_idx : (start_idx + length)] * rescale_factor
        from_grid_mat = from_grid_mat[:, :, self.mapping.coefficient_idx(self.lmax, self.mmax)]

        # save tensors and they will be moved to GPU
        self.register_buffer('to_grid_mat',   to_grid_mat)
        self.register_buffer('from_grid_mat', from_grid_mat)

    def get_to_grid_mat(self):
        # Compute matrices to transform irreps to grid
        return self.to_grid_mat

    def get_from_grid_mat(self):
        # Compute matrices to transform grid to irreps
        return self.from_grid_mat

    def to_grid(self, embedding, lmax, mmax):
        # Compute grid from irreps representation
        to_grid_mat = self.to_grid_mat[:, :, self.mapping.coefficient_idx(lmax, mmax)]
        grid = torch.einsum("bai, zic -> zbac", to_grid_mat, embedding)
        return grid

    def from_grid(self, grid, lmax, mmax):
        # Compute irreps from grid representation
        from_grid_mat = self.from_grid_mat[:, :, self.mapping.coefficient_idx(lmax, mmax)]
        embedding = torch.einsum("bai, zbac -> zic", from_grid_mat, grid)
        return embedding


class SO3_LinearV2(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        lmax,
        bias: bool = True,
        device: str = 'cuda',
    ):
        """
        1. Use `torch.einsum` to prevent slicing and concatenation
        2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        """
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        self.weight = nn.Parameter(torch.randn((self.lmax + 1), out_features, in_features, device=self.device))
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = nn.Parameter(torch.zeros(out_features, device=self.device))

        expand_index = torch.zeros([(lmax + 1) ** 2], device=self.device).long()
        for l in range(lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            expand_index[start_idx: (start_idx + length)] = l
        self.register_buffer('expand_index', expand_index)

    def forward(self, input_embedding):
        weight = torch.index_select(self.weight, dim=0, index=self.expand_index)  # [(L_max + 1) ** 2, C_out, C_in]
        out = torch.einsum('bmi, moi -> bmo', input_embedding.embedding, weight)  # [N, (L_max + 1) ** 2, C_out]
        bias = self.bias.view(1, 1, self.out_features)
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias

        out_embedding = SO3_Embedding(
            0,
            input_embedding.lmax_list.copy(),
            self.out_features,
            device=self.device,
            dtype=input_embedding.dtype
        )
        out_embedding.set_embedding(out)
        out_embedding.set_lmax_mmax(input_embedding.lmax_list.copy(), input_embedding.lmax_list.copy())

        return out_embedding

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax})"


class CoefficientMappingModule(nn.Module):
    """
    Helper module for coefficients used to reshape l <--> m and to get coefficients of specific degree or order

    Args:
        lmax_list (List[int]):   List of maximum degree of the spherical harmonics
        mmax_list (List[int]):   List of maximum order of the spherical harmonics
    """
    def __init__(
        self,
        lmax_list: list,
        mmax_list: list,
        device: str = 'cuda',
    ) -> None:
        super().__init__()
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.device = device

        # Compute the degree (l) and order (m) for each entry of the embedding
        l_harmonic = torch.tensor([], device=self.device).long()
        m_harmonic = torch.tensor([], device=self.device).long()
        m_complex = torch.tensor([], device=self.device).long()

        res_size = torch.zeros([self.num_resolutions], device=self.device).long()
        offset = 0

        for i in range(self.num_resolutions):
            for l in range(0, self.lmax_list[i] + 1):
                mmax = min(self.mmax_list[i], l)
                m = torch.arange(-mmax, mmax + 1, device=self.device).long()
                m_complex = torch.cat([m_complex, m], dim=0)
                m_harmonic = torch.cat(
                    [m_harmonic, torch.abs(m).long()], dim=0
                )
                l_harmonic = torch.cat(
                    [l_harmonic, m.fill_(l).long()], dim=0
                )
            res_size[i] = len(l_harmonic) - offset
            offset = len(l_harmonic)

        num_coefficients = len(l_harmonic)
        # `self.to_m` moves m components from different L to contiguous index
        to_m = torch.zeros([num_coefficients, num_coefficients], device=self.device)
        m_size = torch.zeros([max(self.mmax_list) + 1], device=self.device).long()

        # The following is implemented poorly - very slow. It only gets called
        # a few times so haven't optimized.
        offset = 0
        for m in range(max(self.mmax_list) + 1):
            idx_r, idx_i = self.complex_idx(m, -1, m_complex, l_harmonic)

            for idx_out, idx_in in enumerate(idx_r):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)

            m_size[m] = int(len(idx_r))

            for idx_out, idx_in in enumerate(idx_i):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

        to_m = to_m.detach()

        # save tensors and they will be moved to GPU
        self.register_buffer('l_harmonic', l_harmonic)
        self.register_buffer('m_harmonic', m_harmonic)
        self.register_buffer('m_complex', m_complex)
        self.register_buffer('res_size', res_size)
        self.register_buffer('to_m', to_m)
        self.register_buffer('m_size', m_size)

        # for caching the output of `coefficient_idx`
        self.lmax_cache, self.mmax_cache = None, None
        self.mask_indices_cache = None
        self.rotate_inv_rescale_cache = None

    # Return mask containing coefficients of order m (real and imaginary parts)
    def complex_idx(self, m, lmax, m_complex, l_harmonic):
        """
        Add `m_complex` and `l_harmonic` to the input arguments since we cannot use `self.m_complex`.
        """
        if lmax == -1:
            lmax = max(self.lmax_list)

        indices = torch.arange(len(l_harmonic), device=self.device)

        # Real part
        mask_r = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(m))
        mask_idx_r = torch.masked_select(indices, mask_r)
        mask_idx_i = torch.tensor([], device=self.device).long()

        # Imaginary part
        if m != 0:
            mask_i = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(-m))
            mask_idx_i = torch.masked_select(indices, mask_i)

        return mask_idx_r, mask_idx_i

    # Return mask containing coefficients less than or equal to degree (l) and order (m)
    def coefficient_idx(self, lmax, mmax):
        if (self.lmax_cache is not None) and (self.mmax_cache is not None):
            if (self.lmax_cache == lmax) and (self.mmax_cache == mmax):
                if self.mask_indices_cache is not None:
                    return self.mask_indices_cache

        mask = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_harmonic.le(mmax))
        self.device = mask.device
        indices = torch.arange(len(mask), device=self.device)
        mask_indices = torch.masked_select(indices, mask)
        self.lmax_cache, self.mmax_cache = lmax, mmax
        self.mask_indices_cache = mask_indices
        return self.mask_indices_cache

    # Return the re-scaling for rotating back to original frame
    # this is required since we only use a subset of m components for SO(2) convolution
    def get_rotate_inv_rescale(self, lmax, mmax):
        if (self.lmax_cache is not None) and (self.mmax_cache is not None):
            if (self.lmax_cache == lmax) and (self.mmax_cache == mmax):
                if self.rotate_inv_rescale_cache is not None:
                    return self.rotate_inv_rescale_cache

        if self.mask_indices_cache is None:
            self.coefficient_idx(lmax, mmax)

        rotate_inv_rescale = torch.ones((1, (lmax + 1) ** 2, (lmax + 1) ** 2), device=self.device)
        for l in range(lmax + 1):
            if l <= mmax:
                continue
            start_idx = l ** 2
            length = 2 * l + 1
            rescale_factor = math.sqrt(length / (2 * mmax + 1))
            rotate_inv_rescale[:, start_idx: (start_idx + length), start_idx: (start_idx + length)] = rescale_factor
        rotate_inv_rescale = rotate_inv_rescale[:, :, self.mask_indices_cache]
        self.rotate_inv_rescale_cache = rotate_inv_rescale
        return self.rotate_inv_rescale_cache

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax_list={self.lmax_list}, mmax_list={self.mmax_list})"


class ModuleListInfo(nn.ModuleList):
    def __init__(
        self,
        info_str,
        modules=None,
    ):
        super().__init__(modules)
        self.info_str = str(info_str)

    def __repr__(self):
        return self.info_str


class GraphDropPath(nn.Module):
    """
    Consider batch for graph data when dropping paths.
    """
    def __init__(
        self,
        drop_prob=None,
        device: str = 'cuda',
    ) -> None:
        super(GraphDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.device = device

    def forward(
        self,
        x,
        batch,
    ):
        batch_size = batch.max() + 1
        shape = (batch_size,) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        ones = torch.ones(shape, dtype=x.dtype, device=self.device)
        drop = drop_path(ones, self.drop_prob, self.training)
        out = x * drop[batch]
        return out

    def extra_repr(self):
        return 'drop_prob={}'.format(self.drop_prob)


class EquivariantDropoutArraySphericalHarmonics(nn.Module):
    def __init__(
        self,
        drop_prob,
        drop_graph: bool = False,
        device: str = 'cuda',
    ):
        super(EquivariantDropoutArraySphericalHarmonics, self).__init__()
        self.drop_prob = drop_prob
        self.drop = nn.Dropout(drop_prob, True)
        self.drop_graph = drop_graph
        self.device = device

    def forward(self, x, batch=None):
        if not self.training or self.drop_prob == 0.0:
            return x
        assert len(x.shape) == 3

        if self.drop_graph:
            assert batch is not None
            batch_size = batch.max() + 1
            shape = (batch_size, 1, x.shape[2])
            mask = torch.ones(shape, dtype=x.dtype, device=self.device)
            mask = self.drop(mask)
            out = x * mask[batch]
        else:
            shape = (x.shape[0], 1, x.shape[2])
            mask = torch.ones(shape, dtype=x.dtype, device=self.device)
            mask = self.drop(mask)
            out = x * mask

        return out

    def extra_repr(self):
        return 'drop_prob={}, drop_graph={}'.format(self.drop_prob, self.drop_graph)


class RadialFunction(nn.Module):
    """
    Construct a radial function (lin layers + layer norm + SiLU) given a list of channels
    """
    def __init__(self, channels_list, device: str = 'cuda') -> None:
        super().__init__()
        modules = list()
        input_channels = channels_list[0]
        self.device = device

        for i in range(len(channels_list)):
            if i == 0:
                continue
            modules.append(Linear(input_channels, channels_list[i], bias=True, device=device))
            input_channels = channels_list[i]

            if i == len(channels_list) - 1: break
            modules.append(nn.LayerNorm(channels_list[i], device=device))
            modules.append(nn.SiLU())

        self.net = nn.Sequential(*modules)

    def forward(self, inputs):
        return self.net(inputs)


class ModuleListInfo(nn.ModuleList):
    def __init__(self, info_str, modules=None) -> None:
        super().__init__(modules)
        self.info_str = str(info_str)

    def __repr__(self):
        return self.info_str


class SmoothLeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2

    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)


class GateActivation(nn.Module):
    def __init__(
        self,
        lmax: int,
        mmax: int,
        num_channels: int,
        device: str = 'cuda',
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.num_channels = num_channels
        self.device = device

        # compute `expand_index` based on `lmax` and `mmax`
        num_components = 0
        for l in range(1, self.lmax + 1):
            num_m_components = min((2 * l + 1), (2 * self.mmax + 1))
            num_components = num_components + num_m_components

        expand_index = torch.zeros([num_components], device=device).long()
        start_idx = 0

        for l in range(1, self.lmax + 1):
            length = min((2 * l + 1), (2 * self.mmax + 1))
            expand_index[start_idx: (start_idx + length)] = (l - 1)
            start_idx = start_idx + length

        self.register_buffer('expand_index', expand_index)
        self.scalar_act = nn.SiLU()  # SwiGLU(self.num_channels, self.num_channels)  # #
        self.gate_act = nn.Sigmoid()  # torch.nn.SiLU() # #

    def forward(self, gating_scalars, input_tensors):
        """
        Args:
            gating_scalars: shape [N, lmax * num_channels]
            input_tensors: shape [N, (lmax + 1) ** 2, num_channels]
        """
        gating_scalars = self.gate_act(gating_scalars)
        gating_scalars = gating_scalars.reshape(gating_scalars.shape[0], self.lmax, self.num_channels)
        gating_scalars = torch.index_select(gating_scalars, dim=1, index=self.expand_index)

        input_tensors_scalars = input_tensors.narrow(1, 0, 1)
        input_tensors_scalars = self.scalar_act(input_tensors_scalars)

        input_tensors_vectors = input_tensors.narrow(1, 1, input_tensors.shape[1] - 1)
        input_tensors_vectors = input_tensors_vectors * gating_scalars

        output_tensors = torch.cat((input_tensors_scalars, input_tensors_vectors), dim=1)

        return output_tensors


class S2Activation(nn.Module):
    """
    Assume we only have one resolution
    """
    def __init__(self, lmax, mmax):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.act = nn.SiLU()

    def forward(self, inputs, SO3_grid):
        to_grid_mat = SO3_grid[self.lmax][self.mmax].get_to_grid_mat()  # `device` is not used
        from_grid_mat = SO3_grid[self.lmax][self.mmax].get_from_grid_mat()

        x_grid = torch.einsum("bai, zic -> zbac", to_grid_mat, inputs)
        x_grid = self.act(x_grid)
        outputs = torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)

        return outputs


class SeparableS2Activation(nn.Module):
    def __init__(self, lmax, mmax):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.scalar_act = nn.SiLU()
        self.s2_act = S2Activation(self.lmax, self.mmax)

    def forward(self, input_scalars, input_tensors, SO3_grid):
        output_scalars = self.scalar_act(input_scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[-1])
        output_tensors = self.s2_act(input_tensors, SO3_grid)
        outputs = torch.cat(
            (output_scalars, output_tensors.narrow(1, 1, output_tensors.shape[1] - 1)),
            dim=1,
        )
        return outputs


# Different encodings for the atom distance embeddings
class GaussianSmearing(nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
        device: str = 'cuda',
    ) -> None:
        super(GaussianSmearing, self).__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians, device=device)
        self.coeff = (-0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2)
        self.register_buffer("offset", offset)

    def forward(self, dist) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SigmoidSmearing(nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_sigmoid: int = 50,
        basis_width_scalar: float = 1.0,
        device: str = 'cuda',
    ) -> None:
        super(SigmoidSmearing, self).__init__()
        self.num_output = num_sigmoid
        offset = torch.linspace(start, stop, num_sigmoid, device=device)
        self.coeff = (basis_width_scalar / (offset[1] - offset[0])).item()
        self.register_buffer("offset", offset)

    def forward(self, dist) -> Tensor:
        exp_dist = self.coeff * (dist.view(-1, 1) - self.offset.view(1, -1))
        return torch.sigmoid(exp_dist)


class LinearSigmoidSmearing(nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_sigmoid: int = 50,
        basis_width_scalar: float = 1.0,
        device: str = 'cuda',
    ) -> None:
        super(LinearSigmoidSmearing, self).__init__()
        self.num_output = num_sigmoid
        offset = torch.linspace(start, stop, num_sigmoid, device=device)
        self.coeff = (basis_width_scalar / (offset[1] - offset[0])).item()
        self.register_buffer("offset", offset)
        self.device = device

    def forward(self, dist) -> Tensor:
        exp_dist = self.coeff * (dist.view(-1, 1) - self.offset.view(1, -1))
        x_dist = torch.sigmoid(exp_dist) + 0.001 * exp_dist
        return x_dist


class SiLUSmearing(nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_output: int = 50,
        basis_width_scalar: float = 1.0,
        device: str = 'cuda',
    ) -> None:
        super(SiLUSmearing, self).__init__()
        self.num_output = num_output
        self.fc1 = Linear(2, num_output, device=device)
        self.act = nn.SiLU()
        self.device = device

    def forward(self, dist):
        x_dist = dist.view(-1, 1)
        x_dist = torch.cat([x_dist, torch.ones_like(x_dist)], dim=1)
        x_dist = self.act(self.fc1(x_dist))
        return x_dist


class EquivariantLayerNormArray(nn.Module):
    def __init__(
        self,
        lmax,
        num_channels,
        eps: float = 1e-5,
        affine: bool = True,
        normalization: str = 'component',
        device: str = 'cuda',
    ):
        super().__init__()
        self.device = device
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(lmax + 1, num_channels, device=device))
            self.affine_bias = nn.Parameter(torch.zeros(num_channels, device=device))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """
        out = []

        for l in range(self.lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            feature = node_input.narrow(1, start_idx, length)

            # For scalars, first compute and subtract the mean
            if l == 0:
                feature_mean = torch.mean(feature, dim=2, keepdim=True)
                feature = feature - feature_mean

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == 'norm':
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
            elif self.normalization == 'component':
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)

            if self.affine:
                weight = self.affine_weight.narrow(0, l, 1)  # [1, C]
                weight = weight.view(1, 1, -1)  # [1, 1, C]
                feature_norm = feature_norm * weight  # [N, 1, C]
            feature = feature * feature_norm

            if self.affine and l == 0:
                bias = self.affine_bias
                bias = bias.view(1, 1, -1)
                feature = feature + bias
            out.append(feature)

        out = torch.cat(out, dim=1)

        return out


class EquivariantLayerNormArraySphericalHarmonics(nn.Module):
    """
    1. Normalize over L = 0.
    2. Normalize across all m components from degrees L > 0.
    3. Do not normalize separately for different L (L > 0).
    """
    def __init__(
        self,
        lmax,
        num_channels,
        eps: float = 1e-5,
        affine: bool = True,
        std_balance_degrees: bool = True,
        normalization: str = 'component',
        device: str = 'cuda',
    ):
        super().__init__()
        self.device = device
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees

        # for L = 0
        self.norm_l0 = nn.LayerNorm(self.num_channels, eps=self.eps, elementwise_affine=self.affine, device=device)

        # for L > 0
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.lmax, self.num_channels, device=device))
        else:
            self.register_parameter('affine_weight', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2 - 1, 1, device=device)
            for l in range(1, self.lmax + 1):
                start_idx = l ** 2 - 1
                length = 2 * l + 1
                balance_degree_weight[start_idx: (start_idx + length), :] = (1.0 / length)
            balance_degree_weight = balance_degree_weight / self.lmax
            self.register_buffer('balance_degree_weight', balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees})"

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """
        out = []

        # for L = 0
        feature = node_input.narrow(1, 0, 1)
        feature = self.norm_l0(feature)
        out.append(feature)

        # for L > 0
        if self.lmax > 0:
            num_m_components = (self.lmax + 1) ** 2
            feature = node_input.narrow(1, 1, num_m_components - 1)

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == 'norm':
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
            elif self.normalization == 'component':
                if self.std_balance_degrees:
                    feature_norm = feature.pow(2)  # [N, (L_max + 1)**2 - 1, C], without L = 0
                    feature_norm = torch.einsum('nic, ia -> nac', feature_norm, self.balance_degree_weight)  # [N, 1, C]
                else:
                    feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)

            for l in range(1, self.lmax + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                feature = node_input.narrow(1, start_idx, length)  # [N, (2L + 1), C]
                if self.affine:
                    weight = self.affine_weight.narrow(0, (l - 1), 1)  # [1, C]
                    weight = weight.view(1, 1, -1)  # [1, 1, C]
                    feature_scale = feature_norm * weight  # [N, 1, C]
                else:
                    feature_scale = feature_norm
                feature = feature * feature_scale
                out.append(feature)

        out = torch.cat(out, dim=1)
        return out


class EquivariantRMSNormArraySphericalHarmonics(nn.Module):
    """
    Normalize across all m components from degrees L >= 0.
    """
    def __init__(
        self,
        lmax,
        num_channels,
        eps: float = 1e-5,
        affine: bool = True,
        normalization: str = 'component',
        device: str = 'cuda',
    ):
        super().__init__()
        self.device = device
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        # for L >= 0
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones((self.lmax + 1), self.num_channels, device=device))
        else:
            self.register_parameter('affine_weight', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """
        out = []

        # for L >= 0
        feature = node_input
        if self.normalization == 'norm':
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
        elif self.normalization == 'component':
            feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        for l in range(0, self.lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            feature = node_input.narrow(1, start_idx, length)  # [N, (2L + 1), C]
            if self.affine:
                weight = self.affine_weight.narrow(0, l, 1)  # [1, C]
                weight = weight.view(1, 1, -1)  # [1, 1, C]
                feature_scale = feature_norm * weight  # [N, 1, C]
            else:
                feature_scale = feature_norm
            feature = feature * feature_scale
            out.append(feature)

        out = torch.cat(out, dim=1)
        return out


class EquivariantRMSNormArraySphericalHarmonicsV2(nn.Module):
    """
    1. Normalize across all m components from degrees L >= 0.
    2. Expand weights and multiply with normalized feature to prevent slicing and concatenation.
    """
    def __init__(
        self,
        lmax,
        num_channels,
        eps: float = 1e-5,
        affine: bool = True,
        centering: bool = True,
        std_balance_degrees: bool = True,
        normalization: str = 'component',
        device: str = 'cuda',
    ):
        super().__init__()
        self.device = device
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees

        # for L >= 0
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones((self.lmax + 1), self.num_channels, device=self.device))
            if self.centering:
                self.affine_bias = nn.Parameter(torch.zeros(self.num_channels, device=self.device))
            else:
                self.register_parameter('affine_bias', None)
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

        expand_index = get_l_to_all_m_expand_index(self.lmax, device=self.device)
        self.register_buffer('expand_index', expand_index)

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2, 1, device=self.device)
            for l in range(self.lmax + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                balance_degree_weight[start_idx: (start_idx + length), :] = (1.0 / length)
            balance_degree_weight = balance_degree_weight / (self.lmax + 1)
            self.register_buffer('balance_degree_weight', balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, centering={self.centering}, std_balance_degrees={self.std_balance_degrees})"

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """
        feature = node_input

        if self.centering:
            feature_l0 = feature.narrow(1, 0, 1)
            feature_l0_mean = feature_l0.mean(dim=2, keepdim=True)  # [N, 1, 1]
            feature_l0 = feature_l0 - feature_l0_mean
            feature = torch.cat((feature_l0, feature.narrow(1, 1, feature.shape[1] - 1)), dim=1)

        # for L >= 0
        if self.normalization == 'norm':
            assert not self.std_balance_degrees
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
        elif self.normalization == 'component':
            if self.std_balance_degrees:
                feature_norm = feature.pow(2)  # [N, (L_max + 1)**2, C]
                feature_norm = torch.einsum('nic, ia -> nac', feature_norm, self.balance_degree_weight)  # [N, 1, C]
            else:
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        if self.affine:
            weight = self.affine_weight.view(1, (self.lmax + 1), self.num_channels)  # [1, L_max + 1, C]
            weight = torch.index_select(weight, dim=1, index=self.expand_index)  # [1, (L_max + 1)**2, C]
            feature_norm = feature_norm * weight  # [N, (L_max + 1)**2, C]

        out = feature * feature_norm

        if self.affine and self.centering:
            out[:, 0:1, :] = out.narrow(1, 0, 1) + self.affine_bias.view(1, 1, self.num_channels)

        return out


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L10
# _Jd is a list of tensors of shape (2l+1, 2l+1)
_Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))

# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L37
#
# In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower:
# https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py#L92


def wigner_D(l, alpha, beta, gamma):
    if not l < len(_Jd):
        raise NotImplementedError(f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more")

    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)

    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle, l):
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])

    return M


def drop_path(x, drop_prob: float = 0., training: bool = False, device: str = 'cuda'):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...

    I've opted for changing the layer and argument names to 'drop path' rather than mix DropConnect as a
    layer name and use 'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor

    return output


def get_normalization_layer(
    norm_type,
    lmax,
    num_channels,
    eps: float = 1e-5,
    affine: bool = True,
    normalization: str = 'component',
    device: str = 'cuda',
):
    assert norm_type in ['layer_norm', 'layer_norm_sh', 'rms_norm_sh']
    if norm_type == 'layer_norm':
        norm_class = EquivariantLayerNormArray
    elif norm_type == 'layer_norm_sh':
        norm_class = EquivariantLayerNormArraySphericalHarmonics
    elif norm_type == 'rms_norm_sh':
        norm_class = EquivariantRMSNormArraySphericalHarmonicsV2
    else:
        raise ValueError

    return norm_class(lmax, num_channels, eps, affine, normalization, device=device)


def get_l_to_all_m_expand_index(lmax, device: str = 'cuda'):
    expand_index = torch.zeros([(lmax + 1) ** 2], device=device).long()
    for l in range(lmax + 1):
        start_idx = l ** 2
        length = 2 * l + 1
        expand_index[start_idx : (start_idx + length)] = l

    return expand_index


def init_edge_rot_mat(edge_distance_vec, device: str = 'cuda'):
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0 ** 2, dim=1))

    # Make sure the atoms are far enough apart
    # assert torch.min(edge_vec_0_distance) < 0.0001
    if torch.min(edge_vec_0_distance) < 0.0001:
        print(
            "Error edge_vec_0_distance: {}".format(
                torch.min(edge_vec_0_distance)
            )
        )

    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = edge_vec_2 / (
        torch.sqrt(torch.sum(edge_vec_2 ** 2, dim=1)).view(-1, 1)
    )
    # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
    # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]

    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)

    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    # Check the vectors aren't aligned
    assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z ** 2, dim=1, keepdim=True))
    )
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z ** 2, dim=1)).view(-1, 1)
    )
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (
        torch.sqrt(torch.sum(norm_y ** 2, dim=1, keepdim=True))
    )

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    return edge_rot_mat.detach()
