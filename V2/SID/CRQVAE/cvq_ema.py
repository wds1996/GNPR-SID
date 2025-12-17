import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import kmeans, sinkhorn_algorithm


class CosineVectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=None, sk_iters=100, use_linear=0, use_ema=True, ema_decay=0.95, ema_epsilon=1e-5):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.use_linear = use_linear

        # EMA parameters
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon

        if use_ema:
            # self.beta = beta * 0.4
            self.register_buffer('cluster_size', torch.zeros(n_e))
            self.register_buffer('ema_w', torch.zeros(n_e, e_dim))
            if use_linear == 1:
                self.use_linear = 0
        # end EMA
        
        # 初始化码本
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()
        
        if use_ema:
            self.embedding.weight.requires_grad_(False)


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
        latent_norm = F.normalize(latent, dim=1)
        codebook_norm = F.normalize(codebook, dim=1)
        sim = torch.matmul(latent_norm, codebook_norm.t())  # [B, K]
        distances = 1 - sim  # 越小表示越接近

        # 欧氏距离聚类 Euclidean distance for index selection
        # distances = torch.cdist(latent.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)  # [B, K]

        if use_sk and self.sk_epsilon is not None and self.sk_epsilon > 0:
            d_soft = self.center_distance_for_constraint(distances)
            d_soft = d_soft.double()
            Q = sinkhorn_algorithm(d_soft, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any():
                print("Warning: Sinkhorn returned NaN, falling back to argmin")
                indices = torch.argmin(distances, dim=-1)
            else:
                indices = torch.argmax(Q, dim=-1)
        else:
            indices = torch.argmin(distances, dim=-1)

        # Get codebook vectors 等价于 codebook_vec = codebook[indices]
        codebook_vec = F.embedding(indices, codebook)  # [B, D]

        # 投影量化 Compute projection scalar: w = (x · c) / ||c||^2
        dot_product = torch.sum(latent * codebook_vec, dim=-1, keepdim=True)  # [B, 1]
        norm_sq = torch.sum(codebook_vec * codebook_vec, dim=-1, keepdim=True)
        scalar = dot_product / (norm_sq + 1e-8)                                # [B, 1]
        proj_vec = scalar * codebook_vec
        
        # 直接量化
        # scalar = torch.ones(B, device=x.device) # 为了保持接口一致，返回一个全1的scalar
        # proj_vec = codebook_vec
        

        # === 损失计算：EMA 只保留 commitment loss ===
        if self.use_ema:
            commitment_loss = F.cosine_similarity(proj_vec.detach(), latent, dim=-1)
            loss = self.beta * (1 - commitment_loss).mean()
        else:
            # 余弦相似度量化损失
            commitment_loss = F.cosine_similarity(proj_vec.detach(), latent, dim=-1)
            codebook_loss = F.cosine_similarity(proj_vec, latent.detach(), dim=-1)
            loss = (1 - codebook_loss).mean() + self.beta * (1 - commitment_loss).mean()

            # MSE量化损失
            # commitment_loss = F.mse_loss(proj_vec.detach(), x)
            # codebook_loss = F.mse_loss(proj_vec, x.detach())
            # loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator
        x_q = x + (proj_vec - x).detach()

        # 关键：EMA 更新（仅训练时）
        if self.use_ema and self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.n_e).float()
                cluster_size = one_hot.sum(dim=0)
                self.cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)

                dw = torch.zeros_like(self.ema_w)
                # 关键：对齐 device
                dw.index_add_(0, indices, latent.to(self.ema_w.device))
                self.ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

                # 更新 embedding weight（同步最新 EMA）
                n = self.cluster_size.unsqueeze(1).clamp(min=self.ema_epsilon)
                self.embedding.weight.data.copy_(self.ema_w / n)

                # 死码重置
                # dead_threshold = 1.0
                avg_usage = self.cluster_size.mean()
                dead_threshold = avg_usage * 0.1  # 使用率低于平均10%即重置
                dead_indices = torch.where(self.cluster_size < dead_threshold)[0]
                num_dead = dead_indices.numel()
                if num_dead > 0:
                    if B == 0:
                        raise RuntimeError("Batch size is zero!")
                    if B >= num_dead:
                        sample_indices = torch.randperm(B)[:num_dead]
                    else:
                        sample_indices = torch.randint(0, B, (num_dead,), device=latent.device)
                    replace_samples = latent[sample_indices].to(self.embedding.weight.device)
                    self.embedding.weight.data[dead_indices] = replace_samples
                    self.cluster_size[dead_indices] = 1.0
                    self.ema_w[dead_indices] = replace_samples.to(self.ema_w.device)

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