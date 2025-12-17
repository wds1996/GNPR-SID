import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .mlp import MLPLayers
from .rq import ResidualVectorQuantizer


class CRQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=0.25,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
                 use_linear=0,
        ):
        super(CRQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.use_linear = use_linear

        # 编码器
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn)

        # 残差向量量化器
        self.rq = ResidualVectorQuantizer(
            num_emb_list, e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
            use_linear=self.use_linear
        )

        # 基于下游推荐任务进行损失估计，无需再使用重构损失
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x, use_sk=True):
        x = self.encoder(x)
        x_q, rq_loss, codes = self.rq(x, use_sk=use_sk)
        out = self.decoder(x_q)
        return out, rq_loss, codes

    
    def compute_loss(self, quant_loss, out, xs=None):
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, quant_loss, loss_recon

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        x_q, _, (indices, scalars) = self.rq(x_e, use_sk=use_sk)
        # return indices.cpu(), scalars.cpu()  # [B, L], [B, L]
        x_q_cpu = x_q.cpu()
        indices_cpu = indices.cpu()
        return x_q_cpu, indices_cpu
    
    



    