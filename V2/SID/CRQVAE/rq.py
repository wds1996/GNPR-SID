import torch
import torch.nn as nn
import torch.nn.functional as F
# from .cvq import CosineVectorQuantizer
from .cvq_ema import CosineVectorQuantizer

# from .vq import CosineVectorQuantizer


class ResidualVectorQuantizer(nn.Module):

    def __init__(self, n_e_list, e_dim, sk_epsilons=None, beta=0.25,
                 kmeans_init=False, kmeans_iters=100, sk_iters=100, use_linear=0):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.use_linear = use_linear
        self.vq_layers = nn.ModuleList([
                CosineVectorQuantizer(n_e, e_dim,
                    beta=self.beta,
                    kmeans_init=self.kmeans_init,
                    kmeans_iters=self.kmeans_iters,
                    sk_epsilon=sk_epsilon,
                    sk_iters=sk_iters,
                    use_linear=use_linear
                )
                for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)
            ])


    def forward(self, x, use_sk=True):
        original_shape = x.shape
        if x.ndim == 3:
            B, T, D = x.shape
            x = x.view(-1, D)  # [B*T, D]
        elif x.ndim == 2:
            B, D = x.shape
        else:
            raise ValueError("x must be [B, D] or [B, T, D]")
       
        residual = x
        x_q = torch.zeros_like(x)
        all_losses = []
        all_indices = []
        all_scalars = []

        for quantizer in self.vq_layers:
            x_res, loss, indices, scalar = quantizer(residual, use_sk=use_sk)
            
            x_q = x_q + x_res 
            residual = residual - x_res 

            all_losses.append(loss)
            all_indices.append(indices)
            all_scalars.append(scalar)
        
        x_q = x_q.view(original_shape)
        
        mean_loss = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)      # [B, L]
        all_scalars = torch.stack(all_scalars, dim=-1)      # [B, L]

        if len(original_shape) == 3:
            all_indices = all_indices.view(B, T, -1)     # [B, T, L]
            all_scalars = all_scalars.view(B, T, -1)     # [B, T, L]
        else:
            all_indices = all_indices.view(B, -1)        # [B, L]
            all_scalars = all_scalars.view(B, -1)        # [B, L]

        return x_q, mean_loss, (all_indices, all_scalars)

    @torch.no_grad()
    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)