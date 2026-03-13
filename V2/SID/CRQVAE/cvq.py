import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import kmeans, sinkhorn_algorithm


class CosineVectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=None, sk_iters=100, use_linear=0):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.use_linear = use_linear

        # 初始化码本
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
        codebook = self.embedding.weight
        if self.use_linear:
            codebook = self.codebook_projection(codebook)
        return codebook

    @torch.no_grad()
    def init_emb(self, data):
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x, use_sk=True):
        B, D = x.shape
        latent = x.view(B, D)

        if not self.initted and self.training:
            self.init_emb(latent)

        codebook = self.get_codebook()  # [K, D]

        # 相似度聚类 Cosine similarity for index selection
        latent_norm = F.normalize(latent, dim=-1, eps=1e-8)
        codebook_dir = F.normalize(codebook, dim=-1, eps=1e-8)
        sim = latent_norm @ codebook_dir.t()  # [B, K]

        if use_sk and self.sk_epsilon is not None and self.sk_epsilon > 0:
            distances = 1 - sim
            d_soft = self.center_distance_for_constraint(distances).double()
            Q = sinkhorn_algorithm(d_soft, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any():
                print("Warning: Sinkhorn returned NaN, falling back to argmax(sim)")
                indices = sim.argmax(dim=-1)
            else:
                indices = Q.argmax(dim=-1)
        else:
            indices = sim.argmax(dim=-1)


        direction  = F.embedding(indices, codebook_dir)  # [B, D]

        # 投影量化 Compute projection scalar
        scalar = (latent * direction).sum(dim=-1, keepdim=True)
        scalar = scalar.clamp(min=0.0)                           # 可选
        proj_vec = scalar * direction
        

        commitment_loss = F.mse_loss(proj_vec.detach(), latent)
        codebook_loss = F.mse_loss(proj_vec, latent.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator
        x_q = latent  + (proj_vec - latent).detach()

        indices = indices.view(B)        # [B]
        scalar = scalar.view(B)          # [B]

        return x_q, loss, indices, scalar

    @staticmethod
    def center_distance_for_constraint(distances):
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances
