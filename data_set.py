import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import treat_pic as tr_pic


data_path = './data/indianpinearray.npy'
label_path = './data/IPgt.npy'

class Indiapine_dataset(Dataset):
    """
    for this dataset we can get about 21025 data
    """
    def __init__(self, length=1600):
        self.channel = 200
        self.H = 3
        self.W = 3
        self.length = length

    def __getitem__(self, i):
        i = i % self.length

    def __len__(self):
        return self.length


