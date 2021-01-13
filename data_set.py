import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import treat_pic as tr_pic
import numpy as np

data_path = './data/indianpinearray.npy'
label_path = './data/IPgt.npy'

class Indiapine_dataset(Dataset):
    """
    for this dataset we can get about 21025 data
    """
    def __init__(self, length=1600, start_pos=0):
        self.channel = 200
        self.length = length
        self.start_pos = start_pos
        self.date_pth = data_path
        self.label_pth = label_path
        self.input_array = np.load(data_path)
        self.input_array = np.transpose(self.input_array, (2, 0, 1))
        self.input_label = np.load(label_path)

    def __getitem__(self, i):
        i = i + self.start_pos
        x_pos = i // 145
        y_pos = i % 145
        if x_pos == 0:
            x_pos += 1
        if x_pos == 144:
            x_pos -= 1
        if y_pos == 0:
            y_pos += 1
        if y_pos == 144:
            y_pos -= 1
        data = self.input_array[-1, x_pos-1:x_pos+2, y_pos-1:y_pos+2]
        label = self.input_label[x_pos-1:x_pos+2, y_pos-1:y_pos+2]
        return data, label

    def __len__(self):
        return self.length


