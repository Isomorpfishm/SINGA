import torch
from torch import Tensor
from torch.nn import Module, Linear, Embedding
from torch.nn import functional as F


class AtomEmbedding(Module):
    def __init__(
        self, 
        in_scalar:int,
        in_vector:int,
        out_scalar:int, 
        out_vector:int, 
        vector_normalizer:float = 20.0,
    ) -> None:
        super().__init__()
        assert in_vector == 1
        self.in_scalar = in_scalar
        self.vector_normalizer = vector_normalizer
        self.emb_sca = Linear(in_scalar, out_scalar)
        self.emb_vec = Linear(in_vector, out_vector)

    def forward(self, scalar_input:Tensor, vector_input:Tensor):
        """
        Input:
            scalar_input (Tensor): Shape (*, in_scalar)
            vector_input (Tensor): Shape (*, 3) 
            
        Output: Tuple(Tensor, Tensor)
            sca_emb (Tensor): Shape (scalar_input.shape[0], out_scalar)
            vec_emb (Tensor): Shape (scalar_input.shape[0], out_vector)
        """
        vector_input = vector_input / self.vector_normalizer
        assert vector_input.shape[1:] == (3, ), 'Not support. Only one vector can be input'
        sca_emb = self.emb_sca(scalar_input[:, :self.in_scalar])  # b, f -> b, f'
        vec_emb = vector_input.unsqueeze(-1)  # b, 3 -> b, 3, 1
        vec_emb = self.emb_vec(vec_emb).transpose(1, -1)  # b, 1, 3 -> b, f', 3
        
        return sca_emb, vec_emb 
