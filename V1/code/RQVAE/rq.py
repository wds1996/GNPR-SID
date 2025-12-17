import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):

    def __init__(
            self,
            n_e_list,
            e_dim,
            sk_epsilons,
            kmeans_init=False,
            kmeans_iters=100,
            sk_iters=100,
            use_linear=0,
            use_sk=False,
            beta=0.25,
            diversity_loss=0.0,
    ):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.use_linear = use_linear
        self.use_sk = use_sk
        self.sk_iters = sk_iters
        self.vq_layers = nn.ModuleList([VectorQuantizer(
            n_e,
            e_dim,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilon=sk_epsilon,
            sk_iters=sk_iters,
            use_linear=use_linear,
            use_sk=use_sk,
            beta=beta,
            diversity_loss=diversity_loss,
        ) for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, epoch_idx):
        all_losses = []
        all_indices = []
        all_distances = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices, distance = quantizer(residual, epoch_idx)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)
            all_distances.append(distance)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        all_distances = torch.stack(all_distances, dim=1)

        return x_q, mean_losses, all_indices, all_distances