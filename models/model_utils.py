import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class CoordConv2d(nn.Module):
    """
    CoordConv implementation (from uber, but really from like the 90s)
    """
    def __init__(self, *args, use_coords=False, attend=False, batch_norm=False, dropout=0, conditioning=0, **kwargs):
        super(CoordConv2d, self).__init__()
        self.use_coords = use_coords
        self.attend = attend
        self.conditioning = conditioning
        self.width = 5
        self.height = 5
        self.dropout = nn.Dropout2d(0)
        self.batch_norm = nn.BatchNorm2d(args[1]) if batch_norm else lambda x: x

        coords = torch.zeros(2, self.width, self.height)
        coords[0] = torch.stack([torch.arange(self.height).float()]*self.width, dim=0) * 2 / self.height - 1
        coords[1] = torch.stack([torch.arange(self.width).float()]*self.height, dim=1) * 2 / self.width - 1
        self.register_buffer('coords', coords)

        if self.use_coords:
            args = list(args)
            args[0] += 2
            args = tuple(args)

        self.conv = nn.Conv2d(*args, **kwargs)

        if self.attend:
            self.attend = SpatialAttention2d(args[1], self.attend, self.conditioning)

    def reset(self):
        self.height = 5
        self.width = 5
        self.setup()

    def setup(self):
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width))
        self.coords = torch.zeros(2, self.width, self.height).to(self.coords)
        self.coords[0] = torch.from_numpy(pos_x).float().to(self.coords)
        self.coords[1] = torch.from_numpy(pos_y).float().to(self.coords)

    def forward(self, data, cond=None):
        if self.use_coords:
            flag = False
            if not (self.width == data.shape[2]):
                self.width = data.shape[2]
                flag = True
            if not (self.height == data.shape[3]):
                self.height = data.shape[3]
                flag = True

            if flag:
                self.setup()

            data = torch.cat([data, torch.stack([self.coords]*data.size(0), dim=0) * data.mean() * 2], dim=1)


        x = self.conv(data)
        x = F.leaky_relu(x)
        x2 = None
        if self.attend:
            x2 = self.attend(x, cond)
            x3 = x2 / x
            x = x2
            x2 = x3
        x = self.batch_norm(x)
        if x2 is not None:
            return self.dropout(x), x2
        return self.dropout(x)



class SpatialAttention2d(nn.Module):
    """
    CoordConv implementation (from uber, but really from like the 90s)
    """
    def __init__(self, channels, k, conditioning=0):
        super(SpatialAttention2d, self).__init__()
        self.conditioning = conditioning
        self.lin1 = nn.Linear(channels, k)
        self.lin2 = nn.Linear(k+conditioning, 1)

    def forward(self, data, cond=None, b_print=False, print_path=''):
        orgnl_shape = list(data.shape)
        if (cond is None) or (self.conditioning == 0):
            cond = torch.zeros(orgnl_shape[0], self.conditioning).to(data)
        temp_shape = [orgnl_shape[0], orgnl_shape[2]*orgnl_shape[3], orgnl_shape[1]]
        orgnl_shape[1] = 1
        atn = data.permute(0, 2, 3, 1).view(temp_shape)
        atn = self.lin1(atn)
        atn = torch.tanh(torch.cat([atn, cond.unsqueeze(1).expand(atn.size(0), atn.size(1), self.conditioning)], dim=2))
        atn = self.lin2(atn)
        soft = torch.softmax(atn, dim=1)
        mask = soft.permute(0, 2, 1).view(orgnl_shape)
        if b_print:
            plt.figure(1)
            plt.imshow(mask[0, 0].detach().cpu().numpy())
            plt.savefig(print_path+'mask.png')
        return data*mask


class ChannelAttention2d(nn.Module):
    """
    CoordConv implementation (from uber, but really from like the 90s)
    """
    def __init__(self, img_size):
        super(ChannelAttention2d, self).__init__()
        self.lin = nn.Linear(img_size, 1)

    def forward(self, data):
        atn = self.lin(data.view(data.shape[0], data.shape[1], 1, -1))
        soft = torch.softmax(atn, dim=1)
        mask = soft*atn.shape[1]
        return data#*mask


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Implementation
    """
    def __init__(self):
        super(SpatialSoftmax, self).__init__()
        self.height = 5
        self.width = 5
        self.channel = 5

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width))
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def reset(self):
        self.height = 5
        self.width = 5
        self.channel = 5
        self.setup()

    def setup(self):
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width))
        self.pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).to(self.pos_x).double()
        self.pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).to(self.pos_y).double()

    def forward(self, feature):
        flag = False
        if not (self.channel == feature.shape[1]):
            self.channel = feature.shape[1]
            flag = True
        if not (self.width == feature.shape[2]):
            self.width = feature.shape[2]
            flag = True
        if not (self.height == feature.shape[3]):
            self.height = feature.shape[3]
            flag = True

        if flag:
            self.setup()

        feature = torch.log(F.relu(feature.view(-1, self.height*self.width)) + 1e-6)
        softmax_attention = F.softmax(feature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints
