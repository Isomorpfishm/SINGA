from easydict import EasyDict
from typing import Optional, Union, Dict

import torch
import torch.nn as nn

import torch_geometric as pyg
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

try:
    from EF_layers import (SO3_Embedding,
                           GaussianSmearing,
                           init_edge_rot_mat,
                           SO3_Rotation,
                           EdgeDegreeEmbedding,
                           CoefficientMappingModule,
                           TransBlockV2,
                           ModuleListInfo,
                           SO3_Grid,
                           get_normalization_layer)
except:
    from model.EF_layers import (SO3_Embedding,
                                 GaussianSmearing,
                                 init_edge_rot_mat,
                                 SO3_Rotation,
                                 EdgeDegreeEmbedding,
                                 CoefficientMappingModule,
                                 TransBlockV2,
                                 ModuleListInfo,
                                 SO3_Grid,
                                 get_normalization_layer)

_AVG_NUM_MODES = 77.81317
_AVG_DEGREE = 23.395238876342773

"""
Task to complete:
    0. Jupyter notebook to Python script (here)                             [COMPLETED]
    1. Determine _AVG_NUM_MODES and _AVG_DEGREE
    2. Have external library/pickle files for protein-ligand atomic numbers [COMPLETED]
    3. Export hyperparameters of the embedding layer to the json file
    4. Provide heterogeneous graphs as inputs (pyg.HeteroData)
    5. Direct outputs (SO3_Embedding()) to the GAN model
    
Part of this codebase is adapted from EquiformerV2:
    URL: https://github.com/atomicarchitects/equiformer_v2/tree/main/nets/equiformer_v2
"""


class EquivariantEmbedding(nn.Module):
    """
    Arguments:

    Outputs:

    """
    def __init__(
        self,
        config: EasyDict,
        device: str = 'cuda',
    ) -> None:
        super().__init__()
        self.device = device
        self.edge_channels = config.edge_channels
        self.sphere_channels = config.sphere_channels
        self.attn_hidden_channels = config.attn_hidden_channels
        self.attn_alpha_channels = config.attn_alpha_channels
        self.attn_value_channels = config.attn_value_channels
        self.ffn_hidden_channels = config.ffn_hidden_channels

        self.lmax_list = [int(i) for i in config.lmax_list]
        self.mmax_list = [int(i) for i in config.mmax_list]
        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        self.max_num_elements = config.max_num_elements
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers

        self.offset = 0
        self.offset_res = 0
        self.cutoff = config.cutoff
        self.alpha_drop = config.alpha_drop
        self.proj_drop = config.proj_drop
        self.drop_path_rate = config.drop_path_rate

        self.norm_type = config.norm_type
        self.activation_type = config.activation_type
        self.grid_resolution = None
        self.share_atom_edge_embedding = config.share_atom_edge_embedding
        self.use_atom_edge_embedding = config.use_atom_edge_embedding

        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding

        # Initialise the module that computes WignerD matrices and other values for spherical harmonics calc
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i], device=self.device))

        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all, device=self.device)
        self.sphere_embedding_2 = nn.Embedding(32767, self.sphere_channels_all, device=self.device)

        self.distance_expansion = GaussianSmearing(
            start=0.0,
            stop=self.cutoff,
            num_gaussians=self.edge_channels,
            basis_width_scalar=20,
            device=self.device,
        )
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels] * 2

        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1], device=self.device)
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1], device=self.device)
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None
            
        self.mappingReduced = CoefficientMappingModule(
            self.lmax_list,
            self.mmax_list,
            device=self.device,
        )

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l,
                        m,
                        resolution=self.grid_resolution,
                        normalization='component',
                        device=self.device,
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels=self.sphere_channels,
            lmax_list=self.lmax_list,
            mmax_list=self.mmax_list,
            SO3_rotation=self.SO3_rotation,
            mappingReduced=self.mappingReduced,
            max_num_elements=self.max_num_elements,
            edge_channels_list=self.edge_channels_list,
            use_atom_edge_embedding=False,
            rescale_factor=_AVG_DEGREE,
            device=self.device,
        )

        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                SO3_rotation=self.SO3_rotation,
                SO3_grid=self.SO3_grid,
                mappingReduced=self.mappingReduced,

                sphere_channels=self.sphere_channels,
                attn_hidden_channels=self.attn_hidden_channels,
                attn_alpha_channels=self.attn_alpha_channels,
                attn_value_channels=self.attn_value_channels,
                ffn_hidden_channels=self.ffn_hidden_channels,
                output_channels=self.sphere_channels,

                lmax_list=self.lmax_list,
                mmax_list=self.mmax_list,

                num_heads=self.num_heads,
                max_num_elements=self.max_num_elements,
                edge_channels_list=self.edge_channels_list,
                use_atom_edge_embedding=self.block_use_atom_edge_embedding,

                norm_type=self.norm_type,
                attn_activation=self.activation_type,
                ffn_activation=self.activation_type,
                use_m_share_rad=False,
                use_s2_act_attn=False,
                use_attn_renorm=True,
                use_gate_act=False,
                use_grid_mlp=False,
                use_sep_s2_act=True,

                alpha_drop=self.alpha_drop,
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                device=self.device,
            )
            self.blocks.append(block)

        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=max(self.lmax_list),
            num_channels=self.sphere_channels,
            device=self.device,
        )

    def forward(
        self,
        g: Union[HeteroData, Batch],
        batch: Optional[int] = None,
    ) -> Dict:
        """
        Arguments:
            g (Union[HeteroData, HeteroDataBatch])

        Output:

        """
        x_dict = {}
        if isinstance(g, Batch):
            batch = 64

        """
        Part 1: Protein nodes and edges embedding
            hetero: False [Not considering PL edges]
        """
        hetero = False
        atomic_numbers = g['atomicnum']['protein_atoms']
        protein_x = g['protein_atoms']['x']
        protein_p = g['protein_atoms']['pos']
        protein_ei = g[('protein_atoms', 'linked_to', 'protein_atoms')]['edge_index']
        protein_ev = torch.squeeze(protein_p[protein_ei[0]] - protein_p[protein_ei[1]])
        protein_ed = torch.squeeze(torch.norm(protein_ev, dim=-1, p=2))

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = init_edge_rot_mat(protein_ev)

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        x = SO3_Embedding(
            length=len(atomic_numbers),
            lmax_list=self.lmax_list,
            num_channels=self.sphere_channels,
            device=self.device,
            dtype=torch.long,
        )   # x.embedding[:, offset_res, :] = sphere_embedding(atomic_numbers)

        # One-hot encoding -> barcode, and barcode -> integer for the nn.Embedding() layer
        protein_xtra_feat = list()
        for i in range(protein_x.shape[0]):
            bin_a = ''.join(map(str, protein_x[i, -15:].type(torch.long).tolist()))
            protein_xtra_feat.append(int(bin_a, 2))

        node_features_xtra = self.sphere_embedding_2(
            torch.tensor(
                protein_xtra_feat,
                dtype=torch.long,
                device=self.device,
            )
        )
        x.embedding[:, self.offset_res, :] = self.sphere_embedding(atomic_numbers) + node_features_xtra

        # offset = self.offset + self.sphere_channels
        # offset_res = self.offset_res + int((self.lmax_list[0]+1)**2)

        source_element = atomic_numbers[protein_ei[0]]
        target_element = atomic_numbers[protein_ei[1]]
        source_embedding = self.source_embedding(source_element)
        target_embedding = self.target_embedding(target_element)

        # Edge-degree embedding
        edge_distance = self.distance_expansion(protein_ed)
        edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            protein_ei,
            hetero=hetero,
        )
        x.embedding = x.embedding + edge_degree.embedding

        for i in range(self.num_layers):
            x = self.blocks[i](
                x=x,
                atomic_numbers=atomic_numbers,
                edge_distance=edge_distance,
                edge_index=protein_ei,
                batch=len(atomic_numbers),
                hetero=hetero,
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)
        x_dict['protein_atoms'] = x

        """
        Part 2: Ligand nodes and edges embedding
            hetero: False [Not considering PL edges]
        """
        hetero = False
        atomic_numbers = g['atomicnum']['ligand_atoms']
        ligand_x  = g['ligand_atoms']['x']
        ligand_p  = g['ligand_atoms']['pos']
        ligand_ei = g[('ligand_atoms', 'linked_to', 'ligand_atoms')]['edge_index']
        ligand_ev = torch.squeeze(ligand_p[ligand_ei[0]] - ligand_p[ligand_ei[1]])
        ligand_ed = torch.squeeze(torch.norm(ligand_ev, dim=-1, p=2))

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = init_edge_rot_mat(ligand_ev)

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        x = SO3_Embedding(
            length=len(atomic_numbers),
            lmax_list=self.lmax_list,
            num_channels=self.sphere_channels,
            device=self.device,
            dtype=torch.long,
        )

        ligand_xtra_feat = list()
        for i in range(ligand_x.shape[0]):
            bin_a = ''.join(map(str, ligand_x[i, -15:].type(torch.long).tolist()))
            ligand_xtra_feat.append(int(bin_a, 2))

        node_features_xtra = self.sphere_embedding_2(
            torch.tensor(
                ligand_xtra_feat,
                dtype=torch.long,
                device=self.device,
            )
        )
        x.embedding[:, self.offset_res, :] = self.sphere_embedding(atomic_numbers) + node_features_xtra

        source_element = atomic_numbers[ligand_ei[0]]
        target_element = atomic_numbers[ligand_ei[1]]
        source_embedding = self.source_embedding(source_element)
        target_embedding = self.target_embedding(target_element)

        # Edge-degree embedding
        edge_distance = self.distance_expansion(ligand_ed)
        edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            ligand_ei,
            hetero=hetero,
        )
        x.embedding = x.embedding + edge_degree.embedding

        for i in range(self.num_layers):
            x = self.blocks[i](
                x=x,
                atomic_numbers=atomic_numbers,
                edge_distance=edge_distance,
                edge_index=ligand_ei,
                batch=len(atomic_numbers),
                hetero=hetero,
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)
        x_dict['ligand_atoms'] = x

        """
        Part 3: Ligand-protein edge embedding
            hetero: True [Consider LP edges]
            source: ligand
            target: protein
        """
        hetero = True
        atomic_numbers = dict()
        atomic_numbers['protein_atoms'] = g['atomicnum']['protein_atoms']
        atomic_numbers['ligand_atoms'] = g['atomicnum']['ligand_atoms']

        LP_ei = g[('ligand_atoms', 'interact_with', 'protein_atoms')]['edge_index']
        LP_ev = torch.squeeze(ligand_p[LP_ei[0]] - protein_p[LP_ei[1]])
        LP_ed = torch.squeeze(torch.norm(LP_ev, dim=-1, p=2))

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = init_edge_rot_mat(LP_ev)

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        source_element = atomic_numbers['ligand_atoms'][LP_ei[0]]
        target_element = atomic_numbers['protein_atoms'][LP_ei[1]]
        source_embedding = self.source_embedding(source_element)
        target_embedding = self.target_embedding(target_element)

        # Edge-degree embedding
        edge_distance = self.distance_expansion(LP_ed)
        edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            LP_ei,
            hetero=hetero,
            source_target=('ligand_atoms', 'protein_atoms'),
        )
        x_dict['protein_atoms'].embedding += edge_degree.embedding

        if batch is None:
            batch = len(atomic_numbers['protein_atoms'])

        for i in range(self.num_layers):
            x = self.blocks[i](
                x=x_dict,
                atomic_numbers=atomic_numbers,
                edge_distance=edge_distance,
                edge_index=LP_ei,
                batch=batch,  # batch = len(atomic_numbers['protein_atoms'])
                hetero=True,
                source_target=('ligand_atoms', 'protein_atoms'),
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)
        x_dict['lp_edge'] = x  # x has shape (num_protein_atoms, 49, spherical channels)

        """
        Part 4: Protein-ligand edge embedding
            hetero: True [Consider PL edges]
            source: protein
            target: ligand
        """
        hetero = True
        PL_ei = g[('protein_atoms', 'interact_with', 'ligand_atoms')]['edge_index']
        PL_ev = torch.squeeze(protein_p[PL_ei[0]] - ligand_p[PL_ei[1]])
        PL_ed = torch.squeeze(torch.norm(PL_ev, dim=-1, p=2))

        source_element = atomic_numbers['protein_atoms'][PL_ei[0]]
        target_element = atomic_numbers['ligand_atoms'][PL_ei[1]]
        source_embedding = self.source_embedding(source_element)
        target_embedding = self.target_embedding(target_element)

        # Edge-degree embedding
        edge_distance = self.distance_expansion(PL_ed)
        edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            PL_ei,
            hetero=hetero,
            source_target=('protein_atoms', 'ligand_atoms'),
        )
        x_dict['ligand_atoms'].embedding += edge_degree.embedding

        if batch is None:
            batch = len(atomic_numbers['protein_atoms'])

        for i in range(self.num_layers):
            x = self.blocks[i](
                x=x_dict,
                atomic_numbers=atomic_numbers,
                edge_distance=edge_distance,
                edge_index=PL_ei,
                batch=batch,
                hetero=True,
                source_target=('protein_atoms', 'ligand_atoms'),
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)
        x_dict['pl_edge'] = x  # x has shape (num_ligand_atoms, 49, 16)

        x_dict['protein_atoms'].embedding += x_dict['lp_edge'].embedding
        x_dict['ligand_atoms'].embedding += x_dict['pl_edge'].embedding

        return x_dict
