import numpy as np
import torch
import pandas as pd
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .layers import MLPLayers
from .rq import ResidualVectorQuantizer
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=16,
                 num_emb_list=[32,32,32],
                 e_dim=64,
                 layers=[128,64],
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=20,
                 # sk_epsilons=[0,0,0.003,0.01],
                 sk_epsilons=None,
                 sk_iters=20,
                 use_sk=False,
                 use_linear=0,
                 beta=0.25,
                 diversity_loss=0.0,
                 ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.use_sk = use_sk
        self.use_linear = use_linear
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob, 
                                 bn=self.bn,
                                 weight_init='xavier')

        self.rq = ResidualVectorQuantizer(
            num_emb_list,
            e_dim,
            beta=beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
            use_sk=use_sk,
            use_linear=use_linear,
            diversity_loss=diversity_loss
        )

        self.decode_layer_dims = [self.e_dim] + self.layers + [self.in_dim]
        self.decoder = MLPLayers(
            layers=self.decode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn,
            weight_init='xavier')
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, epoch_idx):
        x = self.encoder(x)
        x_q, rq_loss, indices, distances = self.rq(x, epoch_idx)
        # print(indices.shape)
        out = self.decoder(x_q)
        out = self.sigmoid(out)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, epoch_idx=0):
        x_e = self.encoder(xs)
        x_q, _, indices, distances = self.rq(x_e, epoch_idx)
        return x_q, indices, distances

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            # loss_recon = F.mse_loss(out, xs, reduction='sum')/xs.size(0)
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            # loss_recon = F.l1_loss(out, xs, reduction='sum')/xs.size(0)
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon
    