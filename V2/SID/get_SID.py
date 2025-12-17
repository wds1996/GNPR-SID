import torch
from torch.utils.data import DataLoader
import pandas as pd
import csv
from collections import Counter
import os
import argparse
import random
import numpy as np
import logging
from tqdm import tqdm
import datetime
from POIdatasets import EmbDataset
from CRQVAE.crqvae import CRQVAE

def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)

def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur

def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def parse_args(datafold):
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=3000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=10, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="constant", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=100, help='warmup epochs')
    parser.add_argument("--data_path", type=str, default=f"", help="Path to POI embedding dict (.pkl)")

    parser.add_argument("--weight_decay", type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=True, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--use_sk', type=bool, default=False, help="use sinkhorn or not")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.1, 0.1, 0.1], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    parser.add_argument("--use-linear", type=int, default=1, help="use-linear")

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[64,64,64], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=64, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=0.5, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[512,256,128], help='hidden sizes of every layer')

    parser.add_argument('--save_limit', type=int, default=5)
    parser.add_argument("--ckpt_dir", type=str, default=f"", help="output directory for model")

    return parser.parse_args()


def get_quantization():
    
    """fix the random seed"""
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    datafold = ""
    args = parse_args(datafold)
    print("=================================================")
    print(args)
    print("=================================================")

    logging.basicConfig(level=logging.DEBUG)

    """build dataset"""
    data = EmbDataset(args.data_path)

    model = CRQVAE(in_dim=data.dim,
                   num_emb_list=args.num_emb_list,
                   e_dim=args.e_dim,
                   layers=args.layers,
                   dropout_prob=args.dropout_prob,
                   bn=args.bn,
                   loss_type=args.loss_type,
                   quant_loss_weight=args.quant_loss_weight,
                   beta=args.beta,
                   kmeans_init=args.kmeans_init,
                   kmeans_iters=args.kmeans_iters,
                   sk_epsilons=args.sk_epsilons,
                   sk_iters=args.sk_iters,
                   use_linear=args.use_linear,
                  )
    # print(model)
    data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)
    
    best_loss_ckpt = "best_loss_model.pth"
    best_collision_ckpt = "best_collision_model.pth"
    time_dir = ""
    best_loss_ckpt_file = args.ckpt_dir + f"{time_dir}/{best_loss_ckpt}"
    best_collision_ckpt_file = args.ckpt_dir + f"{time_dir}/{best_collision_ckpt}"
    
    checkpoint = torch.load(best_loss_ckpt_file, map_location=args.device, weights_only=False)
    
    model = CRQVAE(in_dim=data.dim,
                   num_emb_list=args.num_emb_list,
                   e_dim=args.e_dim,
                   layers=args.layers,
                   dropout_prob=args.dropout_prob,
                   bn=args.bn,
                   loss_type=args.loss_type,
                   quant_loss_weight=args.quant_loss_weight,
                   beta=args.beta,
                   kmeans_init=args.kmeans_init,
                   kmeans_iters=args.kmeans_iters,
                   sk_epsilons=args.sk_epsilons,
                   sk_iters=args.sk_iters,
                   use_linear=args.use_linear,
                  )

    # 加载权重
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(args.device)
    model.eval()
    
    SIDs = {}
    vectors = {}

    iter_data = tqdm(
                data_loader,
                total=len(data_loader),
                ncols=100,
                desc=set_color(f"Generate codebooks ", "pink"),
                )
    
    for batch_idx, data in enumerate(iter_data):
            pids, data = data[0], data[1]
            pids = pids.tolist()
            data = data.to(args.device)
            vector, indices = model.get_indices(data)
            for indx, poi in enumerate(pids):
                SIDs[poi] = indices[indx].tolist()
                vectors[poi] = vector[indx].tolist()

    # print(SIDs)

    value_counts = Counter(tuple(value) for value in SIDs.values())
    seen_values = {}

    updated_dict = {}
    # for key, value in SIDs.items():
    for key in sorted(SIDs.keys()):
        value = SIDs[key]
        value_tuple = tuple(value)  
        if value_counts[value_tuple] > 1: 
            if value_tuple not in seen_values:
                seen_values[value_tuple] = 0
            else:
                seen_values[value_tuple] += 1
            updated_dict[key] = value + [seen_values[value_tuple]]
        else:
            updated_dict[key] = value 

    csv_file = f""
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        writer.writerow(["pid", "sid", "vector"])

        for key, value in updated_dict.items():
            writer.writerow([key, value, vectors[key]])

if __name__ == "__main__":
    get_quantization()