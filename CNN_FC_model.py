import torch
import torchvision
import torch.nn as nn
from torch import optim
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN3Net_224(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3D = nn.Conv3d(1, 20, kernel_size=(24, 3, 3))
        self.Act_F = nn.PReLU()
        self.FC1 = nn.Sequential(nn.Linear(73660, 10000), nn.Dropout(p=0.5))
        self.FC2 = nn.Sequential(nn.Linear(10000, 15), nn.Dropout(p=0.5))

    def forward(self, x):
        # input shape is batch_size * 224 * 3 * 3
        b_size = x.size(0)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # reshape 3*3 into 6*6
        x = x.view(-1, 1, 224, 6, 6)
        conv_res = self.conv3D(x)
        pool1 = F.max_pool3d(conv_res, 1)
        pool2 = F.max_pool3d(conv_res, 2)
        pool3 = F.max_pool3d(conv_res, 3)
        pool1 = pool1.view(b_size, -1)
        pool2 = pool2.view(b_size, -1)
        pool3 = pool3.view(b_size, -1)
        vec_to_FC = torch.cat((pool1, pool2, pool3), dim=1)
        vec_to_FC = self.Act_F(vec_to_FC)
        # print(vec_to_FC.size())
        output = F.relu(self.FC1(vec_to_FC))
        output = self.FC2(output)
        output = F.softmax(output, dim=1)
        # print(output.size())
        return output

class CNN3Net_102(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3D = nn.Conv3d(1, 20, kernel_size=(24, 3, 3))
        self.Act_F = nn.PReLU()
        self.FC1 = nn.Sequential(nn.Linear(28920, 10000), nn.Dropout(p=0.5))
        self.FC2 = nn.Sequential(nn.Linear(10000, 15), nn.Dropout(p=0.5))

    def forward(self, x):
        # input shape is batch_size * 102 * 3 * 3
        b_size = x.size(0)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # reshape 3*3 into 6*6
        x = x.view(-1, 1, 102, 6, 6)
        conv_res = self.conv3D(x)
        pool1 = F.max_pool3d(conv_res, 1)
        pool2 = F.max_pool3d(conv_res, 2)
        pool3 = F.max_pool3d(conv_res, 3)
        pool1 = pool1.view(b_size, -1)
        pool2 = pool2.view(b_size, -1)
        pool3 = pool3.view(b_size, -1)
        vec_to_FC = torch.cat((pool1, pool2, pool3), dim=1)
        vec_to_FC = self.Act_F(vec_to_FC)
        # print(vec_to_FC.size())
        output = F.relu(self.FC1(vec_to_FC))
        output = self.FC2(output)
        output = F.softmax(output, dim=1)
        # print(output.size())
        return output




