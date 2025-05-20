import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
current_dir = os.getcwd()

# Pid,Uid,Catname,Region,Time,neighbors,forward_neighbors

class EmbDataset(Dataset):

    def __init__(self, datapath):
        data = pd.read_csv(current_dir + datapath)
        self.ids = data['Pid']
        data['Uid'] = data['Uid'].apply(eval)
        data['Time'] = data['Time'].apply(eval)
        data['neighbors'] = data['neighbors'].apply(eval)
        data['forward_neighbors'] = data['forward_neighbors'].apply(eval)

        mode = datapath.split('/')[2]
        time_num = 24
        if mode == 'NYC':
            cat_num = 210
            region_num = 92
            neighbor_num = 1084
        elif mode == 'TKY':
            cat_num = 191
            region_num = 60
            neighbor_num = 2294
        elif mode == 'CA':
            cat_num = 304
            region_num = 958
            neighbor_num = 6593
        else:
            raise ValueError("Invalid data mode. Choose from 'NYC', 'TKY', or 'CA'.")


        def to_one_hot_fixed_dim(indices, num_classes, scale_factor=1):
            one_hot = torch.zeros(num_classes, dtype=torch.float32)
            one_hot[indices] = 1
            one_hot *= scale_factor
            return one_hot
        
        catgories =[]
        for cat in data[f'Catname']:  
            cat = to_one_hot_fixed_dim(cat, cat_num, scale_factor=1) 
            catgories.append(cat)
        self.catgorie = catgories

        regions =[]
        for region in data[f'Region']:  
            region = to_one_hot_fixed_dim(region, region_num, scale_factor=1) 
            regions.append(region)
        self.regions = regions

        times =[]
        for time in data[f'Time']:  
            # if len(time) > 10:
            #     time = time[:10]
            time = to_one_hot_fixed_dim(time, time_num, scale_factor=1) 
            times.append(time)
        self.times = times
        
        neighbors = []
        for neighbor in data[f'Uid']:
            # if len(neighbor) > 10:
            #     neighbor = neighbor[:10]
            neighbor = to_one_hot_fixed_dim(neighbor, neighbor_num, scale_factor=1)
            neighbors.append(neighbor)
        self.neighbors = neighbors


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], torch.cat([self.catgorie[idx], self.regions[idx], self.times[idx], self.neighbors[idx]])

