import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):

    def __init__(
            self,
            n_e,
            e_dim,
            beta=0.25,
            kmeans_init=False,
            kmeans_iters=10,
            sk_epsilon=0.01,
            sk_iters=100,
            use_linear=0,
            use_sk=False,
            diversity_loss=0.0
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.use_linear = use_linear
        self.use_sk = use_sk
        self.diversity_loss = diversity_loss

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

        if use_linear == 1:
            self.codebook_projection = torch.nn.Linear(self.e_dim, self.e_dim)
            torch.nn.init.normal_(self.codebook_projection.weight, std=self.e_dim ** -0.5)

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, epoch_idx):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        if self.use_linear == 1:
            embeddings_weight = self.codebook_projection(self.embedding.weight)
        else:
            embeddings_weight = self.embedding.weight

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent ** 2, dim=1, keepdim=True) + \
            torch.sum(embeddings_weight ** 2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, embeddings_weight.t())
        if not self.use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
            # print(Q.sum(0)[:10])
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)


        if self.use_linear == 1:
            x_q = F.embedding(indices, embeddings_weight).view(x.shape)
        else:
            x_q = self.embedding(indices).view(x.shape)

        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())

        if epoch_idx >= 1000:        
            if self.diversity_loss > 0:
                sample_counts = torch.bincount(indices.view(-1), minlength=self.n_e)
                mean_count = indices.size(0) / self.n_e
                mean_count_loss = torch.mean((sample_counts - mean_count) ** 2) / (mean_count ** 2 + 1e-5)
                # pairwise
                # pairwise_loss = 0
                # for i in range(self.n_e):
                #     codebook_vectors = x_q[indices == i]
                #     if len(codebook_vectors) > 1:
                #         pairwise_distances = torch.cdist(codebook_vectors, codebook_vectors, p=2)
                #         pairwise_loss += pairwise_distances.mean()

                diversity_loss = (
                    # 0.1 * (pairwise_loss / self.n_e) +
                    0.1 * mean_count_loss
                )
                loss = codebook_loss + self.beta * commitment_loss + self.diversity_loss * diversity_loss
            else:
                loss = codebook_loss + self.beta * commitment_loss
        else:
            loss = codebook_loss + self.beta * commitment_loss
        
        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices, d