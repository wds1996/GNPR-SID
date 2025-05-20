import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from RQVAE.rqvae import RQVAE
from POIdataset import EmbDataset
import os
import random
import argparse
from trainer import Trainer
import numpy as np
import logging

def parse_args(data_mode):

    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=3000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, )
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="constant", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--data_path", type=str, default=f"/datasets/{data_mode}/poi_info.csv", help="Input data path.")

    parser.add_argument("--weight_decay", type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type") # mse, l1
    parser.add_argument("--kmeans_init", type=bool, default=False, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--use_sk', type=bool, default=False, help="use sinkhorn or not")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.003], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    parser.add_argument("--use-liner", type=int, default=0, help="use-liner")

    parser.add_argument("--device", type=str, default="cuda:7", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[32,32,32], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=64, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument("--lamda", type=float, default=0, help="Lamda for diversity loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128],
                        help='hidden sizes of every layer')

    parser.add_argument('--save_limit', type=int, default=5)
    parser.add_argument("--ckpt_dir", type=str, default="save", help="output directory for model")

    return parser.parse_args()



if __name__ == '__main__':
    """fix the random seed"""
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    data_mode = "TKY"
    args = parse_args(data_mode)
    print("=================================================")
    print(args)
    print("=================================================")
    logging.basicConfig(level=logging.DEBUG)
    """build dataset"""
    data = EmbDataset(args.data_path)
    input_dim = data[0][1].shape[0]
    model = RQVAE(
            in_dim=input_dim, 
            num_emb_list=args.num_emb_list, 
            e_dim=args.e_dim,
            layers=args.layers,
            dropout_prob=args.dropout_prob,
            bn=args.bn,
            loss_type=args.loss_type,
            quant_loss_weight=args.quant_loss_weight,
            kmeans_init=args.kmeans_init,
            kmeans_iters=args.kmeans_iters,
            sk_epsilons=args.sk_epsilons,
            sk_iters=args.sk_iters,
            use_linear=args.use_liner,
            use_sk=args.use_sk,
            beta=args.beta,
            diversity_loss=args.lamda,
    )
    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)

    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss", best_loss)
    print("Best Collision Rate", best_collision_rate)
