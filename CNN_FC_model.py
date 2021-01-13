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
# DEVICE = torch.device("cpu")

class attention_FC224(nn.Module):
    def __init__(self, channel=224, down_scale=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel // 16, 1, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // 16, channel // down_scale, 1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        """
        :param x: a tensor which size is n*224*3*3
        :return: x with self-attention
        """
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return out

class attention_FC102(nn.Module):
    def __init__(self, channel=102, down_scale=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel // 16, 1, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // 16, channel // down_scale, 1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        """
        :param x: a tensor which size is n*224*3*3
        :return: x with self-attention
        """
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return out

class attention_FC200(nn.Module):
    def __init__(self, channel=200, down_scale=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel // 16, 1, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // 16, channel // down_scale, 1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        """
        :param x: a tensor which size is n*224*3*3
        :return: x with self-attention
        """
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return out

class CNN3Net_224(nn.Module):
    def __init__(self, active_fc='PReLU', pretrained=False, model_pth=None):
        super().__init__()
        self.attention_conv = attention_FC224()
        self.conv3D = nn.Conv3d(1, 20, kernel_size=(24, 3, 3), padding=(0, 1, 1))
        self.bn = nn.BatchNorm3d(20)
        if pretrained and model_pth != None:
            model_pre = torch.load(model_pth, map_location='cuda')
            self.conv3D.load_state_dict(model_pre)
        if active_fc == 'PReLU':
            self.Act_F = nn.PReLU()
        elif active_fc == 'CELU':
            self.Act_F = nn.CELU()
        elif active_fc == 'ELU':
            self.Act_F = nn.ELU()
        elif active_fc == 'SELU':
            self.Act_F = nn.SELU()
        self.FC1 = nn.Sequential(nn.Linear(6020, 1000), nn.Dropout(p=0.5))
        self.FC2 = nn.Sequential(nn.Linear(1000, 15), nn.Dropout(p=0.5))

    def forward(self, x):
        # input shape is batch_size * 224 * 3 * 3
        b_size = x.size(0)
        x = self.attention_conv(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # reshape 3*3 into 6*6
        x = x.view(-1, 1, 74, 6, 6)
        conv_res = self.conv3D(x)
        conv_res = self.bn(conv_res)
        # print(conv_res.size())
        pool1 = F.max_pool3d(conv_res, 6)
        pool2 = F.max_pool3d(conv_res, 3)
        pool3 = F.max_pool3d(conv_res, 2)
        pool1 = pool1.view(b_size, -1)
        pool2 = pool2.view(b_size, -1)
        pool3 = pool3.view(b_size, -1)
        vec_to_FC = torch.cat((pool1, pool2, pool3), dim=1)
        vec_to_FC = self.Act_F(vec_to_FC)
        # print(vec_to_FC.size())
        output = F.relu(self.FC1(vec_to_FC))
        output = self.FC2(output)
        output = F.softmax(output, dim=1)
        return output

class CNN3Net_102(nn.Module):
    def __init__(self, active_fc='PReLU', pretrained=False, model_pth=None):
        super().__init__()
        self.attention_conv = attention_FC102()
        self.conv3D = nn.Conv3d(1, 20, kernel_size=(24, 3, 3), padding=(0, 1, 1))
        self.bn = nn.BatchNorm3d(20)
        if pretrained and model_pth != None:
            model_pre = torch.load(model_pth, map_location='cuda')
            self.conv3D.load_state_dict(model_pre)
        if active_fc == 'PReLU':
            self.Act_F = nn.PReLU()
        elif active_fc == 'CELU':
            self.Act_F = nn.CELU()
        elif active_fc == 'ELU':
            self.Act_F = nn.ELU()
        elif active_fc == 'SELU':
            self.Act_F = nn.SELU()
        self.FC1 = nn.Sequential(nn.Linear(1160, 1000), nn.Dropout(p=0.5))
        self.FC2 = nn.Sequential(nn.Linear(1000, 15), nn.Dropout(p=0.5))

    def forward(self, x):
        # input shape is batch_size * 102 * 3 * 3
        b_size = x.size(0)
        x = self.attention_conv(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # reshape 3*3 into 6*6
        x = x.view(-1, 1, 34, 6, 6)
        conv_res = self.conv3D(x)
        conv_res = self.bn(conv_res)
        pool1 = F.max_pool3d(conv_res, 6)
        pool2 = F.max_pool3d(conv_res, 3)
        pool3 = F.max_pool3d(conv_res, 2)
        pool1 = pool1.view(b_size, -1)
        pool2 = pool2.view(b_size, -1)
        pool3 = pool3.view(b_size, -1)
        vec_to_FC = torch.cat((pool1, pool2, pool3), dim=1)
        vec_to_FC = self.Act_F(vec_to_FC)
        # print(vec_to_FC.size())
        output = F.relu(self.FC1(vec_to_FC))
        output = self.FC2(output)
        output = F.softmax(output, dim=1)
        return output

class CNN3Net_200(nn.Module):
    def __init__(self, active_fc='PReLU', pretrained=False, model_pth=None):
        super().__init__()
        self.attention_conv = attention_FC200()
        self.conv3D = nn.Conv3d(1, 20, kernel_size=(24, 3, 3), padding=(0, 1, 1))
        self.bn = nn.BatchNorm3d(20)
        if pretrained and model_pth != None:
            model_pre = torch.load(model_pth, map_location='cuda')
            self.conv3D.load_state_dict(model_pre)
        if active_fc == 'PReLU':
            self.Act_F = nn.PReLU()
        elif active_fc == 'CELU':
            self.Act_F = nn.CELU()
        elif active_fc == 'ELU':
            self.Act_F = nn.ELU()
        elif active_fc == 'SELU':
            self.Act_F = nn.SELU()
        self.FC1 = nn.Sequential(nn.Linear(5040, 1000), nn.Dropout(p=0.5))
        self.FC2 = nn.Sequential(nn.Linear(1000, 17), nn.Dropout(p=0.5))

    def forward(self, x):
        # input shape is batch_size * 224 * 3 * 3
        b_size = x.size(0)
        x = self.attention_conv(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # reshape 3*3 into 6*6
        x = x.view(-1, 1, 66, 6, 6)
        conv_res = self.conv3D(x)
        conv_res = self.bn(conv_res)
        # print(conv_res.size())
        pool1 = F.max_pool3d(conv_res, 6)
        pool2 = F.max_pool3d(conv_res, 3)
        pool3 = F.max_pool3d(conv_res, 2)
        pool1 = pool1.view(b_size, -1)
        pool2 = pool2.view(b_size, -1)
        pool3 = pool3.view(b_size, -1)
        vec_to_FC = torch.cat((pool1, pool2, pool3), dim=1)
        vec_to_FC = self.Act_F(vec_to_FC)
        # print(vec_to_FC.size())
        output = F.relu(self.FC1(vec_to_FC))
        output = self.FC2(output)
        output = F.softmax(output, dim=1)
        return output

class Orin_CNN3Net_200(nn.Module):
    def __init__(self, active_fc='PReLU', pretrained=False, model_pth=None):
        super().__init__()
        self.conv3D = nn.Conv3d(1, 20, kernel_size=(24, 3, 3), padding=(0, 1, 1))
        self.bn = nn.BatchNorm3d(20)
        if pretrained and model_pth != None:
            model_pre = torch.load(model_pth, map_location='cuda')
            self.conv3D.load_state_dict(model_pre)
        if active_fc == 'PReLU':
            self.Act_F = nn.PReLU()
        elif active_fc == 'CELU':
            self.Act_F = nn.CELU()
        elif active_fc == 'ELU':
            self.Act_F = nn.ELU()
        elif active_fc == 'SELU':
            self.Act_F = nn.SELU()
        self.FC1 = nn.Sequential(nn.Linear(21140, 10000), nn.Dropout(p=0.5))
        self.FC2 = nn.Sequential(nn.Linear(10000, 17), nn.Dropout(p=0.5))

    def forward(self, x):
        # input shape is batch_size * 224 * 3 * 3
        b_size = x.size(0)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # reshape 3*3 into 6*6
        x = x.view(-1, 1, 200, 6, 6)
        conv_res = self.conv3D(x)
        conv_res = self.bn(conv_res)
        # print(conv_res.size())
        pool1 = F.max_pool3d(conv_res, 6)
        pool2 = F.max_pool3d(conv_res, 3)
        pool3 = F.max_pool3d(conv_res, 2)
        pool1 = pool1.view(b_size, -1)
        pool2 = pool2.view(b_size, -1)
        pool3 = pool3.view(b_size, -1)
        vec_to_FC = torch.cat((pool1, pool2, pool3), dim=1)
        vec_to_FC = self.Act_F(vec_to_FC)
        # print(vec_to_FC.size())
        output = F.relu(self.FC1(vec_to_FC))
        output = self.FC2(output)
        output = F.softmax(output, dim=1)
        return output

