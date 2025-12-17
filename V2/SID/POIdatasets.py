import numpy as np
import torch
import torch.utils.data as data
import pickle

class EmbDataset(data.Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            emb_dict = pickle.load(f)

        self.ids = sorted(emb_dict.keys()) 
        self.embeddings = np.array([emb_dict[k] for k in self.ids])
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        id = self.ids[index]
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return id, tensor_emb

    def __len__(self):
        return len(self.ids)
