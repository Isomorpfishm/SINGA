from typing import List
import torch
import torch.nn as nn

try:
    from EF_layers import *
except:
    from model.EF_layers import *

_AVG_NUM_MODES = 77.81317
_AVG_DEGREE = 23.395238876342773


class EquiformerV2_Embedding(nn.Module):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:

    """
    def __init__(
        self,
        edge_channels: int = 128,
        sphere_channels: int = 128,
        attn_hidden_channels: int = 128,
        attn_alpha_channels: int = 32,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 512,

        num_heads: int = 8,
        num_layers: int = 12,
        max_num_elements: int = 90,
        max_neighbors: int = 6,
        max_radius: float = 5.0,

        lmax_list: list = [6],
        mmax_list: list = [2],

        share_atom_edge_embedding: bool = True,
        use_atom_edge_embedding: bool = True,

        distance_function: str = 'gaussian',
        grid_resolution: int = None,
        norm_type: str = 'rms_norm_sh',

        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.05,
        proj_drop: float = 0.0,
        device: str = 'cuda',
    ):
        super().__init__()
        self.device = device
        self.edge_channels = edge_channels
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_num_elements = max_num_elements
        self.max_radius = max_radius
        self.max_neighbors = max_neighbors
        self.cutoff = max_radius

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list

        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all)
        self.share_atom_edge_embedding = share_atom_edge_embedding
        self.use_atom_edge_embedding = use_atom_edge_embedding

        self.distance_function = distance_function
        self.grid_resolution = grid_resolution
        self.num_resolutions = len(self.lmax_list)
        self.norm_type = norm_type

        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        if self.distance_function == 'gaussian':
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                600,
                2.0,
            )
        else:
            raise ValueError("Only gaussian is accepted as of now")

        # Initialise the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels] * 2

        # Initialise the atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1], device=self.device)
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1], device=self.device)
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None

        # Initialise the module that computes WignerD matrices and other values for spherical harmonics calc
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialise conversion between degree 1 and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialise the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list)+1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(SO3_Grid(l, m, resolution=self.grid_resolution, normalization='component'))
            self.SO3_grid.append(SO3_m_grid)

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=_AVG_DEGREE,
        )

        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop,
                self.drop_path_rate,
                self.proj_drop,
            )
            self.blocks.append(block)

    def forward(self, data):
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype

        atomic_numbers = data.atomic_numbers.long()
        num_atoms = len(atomic_numbers)
        pos = data.pos

        (edge_index, edge_distance, edge_distance_vec, cell_offsets, _, neighbors) = self.generate_graph(data)

        #############################
        # Initialise data structures
        #############################

        # Compute 3 x 3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(data, edge_index, edge_distance_vec)

        # Initialise the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        #############################
        # Initialise node embeddings
        #############################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res, offset = 0, 0
        # Initialise the l=0, m=0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[:, offset:offset+self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i]+1)**2)

        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]
            target_element = atomic_numbers[edge_index[1]]
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            edge_index,
        )
        x.embedding = x.embedding + edge_degree.embedding

        ##################################
        # Update spherical node embeddings
        ##################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=data.batch
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        return x

    # Initialise the edge rotation matrices
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)
