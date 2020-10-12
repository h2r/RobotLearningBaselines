import torch.nn as nn
import torch
from utils.math import *
from models.model_utils import *


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128,128,128), activation='tanh', log_std=0):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh
            self.affine_layers.append(nn.Dropout(.5))


        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        # Here, we're just normalizing the Quaternion to be unit magnitude so that 
        # RLBench's motion planner doesn't complain
        norma = torch.tanh(action_mean[:, 3:7])
        norm = norma*norma
        norm = norm.sum(dim=1, keepdim=True)
        norm = norm.sqrt()
        action_mean = torch.cat([action_mean[:, :3], norma/norm, torch.sigmoid(action_mean[:, 7:])], dim=1)

        return action_mean, action_log_std, action_std

    def select_action(self, x):
        action_mean, _, action_std = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action

    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)

        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}



class VisionPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128,128,128)):
        super().__init__()
        self.activation = nn.ReLU()
        
        self.conv1 = CoordConv2d(3,   32,  kernel_size=5, stride=2, batch_norm=True, attend=0,   use_coords=False, dropout=.2)
        self.conv2 = CoordConv2d(32,  32,  kernel_size=3, stride=1, batch_norm=True, attend=0,   use_coords=False, dropout=.2)
        self.conv3 = CoordConv2d(32,  64,  kernel_size=3, stride=2, batch_norm=True, attend=0,   use_coords=False, dropout=.2)
        self.conv4 = CoordConv2d(64,  64,  kernel_size=3, stride=1, batch_norm=True, attend=0,   use_coords=False, dropout=.2)
        self.conv5 = CoordConv2d(64,  128, kernel_size=3, stride=2, batch_norm=True, attend=512, use_coords=True,  dropout=.2)
        self.conv6 = CoordConv2d(128, 128, kernel_size=3, stride=1, batch_norm=True, attend=512, use_coords=True,  dropout=.2)
        self.conv7 = CoordConv2d(128, 256, kernel_size=3, stride=1, batch_norm=True, attend=512, use_coords=True,  dropout=.2)
        self.conv8 = CoordConv2d(256, 256, kernel_size=3, stride=1, batch_norm=True, attend=512, use_coords=False, dropout=.2)

        self.conv_lin1 = nn.Linear(256*(7*7), 512)
        self.conv_lin2 = nn.Linear(512, 512)
        self.dropout   = nn.Dropout(.5)

        self.spatial_softmax = SpatialSoftmax()

        self.resetables = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, self.spatial_softmax]

        '''
        self.aux = nn.Sequential(nn.Linear(512*2 + tau_size, 512, bias=use_bias),
                                 nn.LeakyReLU(),
                                 nn.Linear(512, aux_size, bias=use_bias))
        '''

        self.fl1 = nn.Linear(512*2 + state_dim[0], 512)# + tau_size + aux_size
        self.fl2 = nn.Linear(512, 512)
        self.output = nn.Linear(512, action_dim + 6)


    def A_vec_to_quat(self, A_vec):
        A = torch.zeros((A_vec.shape[0], 4, 4)).to(A_vec)
        A[:, 0, :] = A_vec[:, :4]
        A[:, :, 0] = A_vec[:, :4]
        for i in range(1, 4):
            for j in range(1, 4):
                A[:, i, j] = A_vec[:, i + j + 3]
        evs = torch.stack([torch.symeig(A[i], eigenvectors=True)[1][:, 0] for i in range(A.shape[0])])
        return evs

    def forward(self, low_dim, vision, b_print=False):
        x = self.conv1(vision)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x, x5 = self.conv5(x)

        # if b_print:
        #     plt.figure(1)
        #     for i in range(min(64, x.size(1))):
        #         plt.subplot(8,8,i+1)
        #         try:
        #             plt.imshow(x[0,i].detach().cpu().numpy())
        #         except:
        #             y = x5[0,i].detach().cpu().numpy()
        #             plt.imshow(y)
        #     plt.savefig('activations5.png')

        x, x6 = self.conv6(x)

        # if b_print:
        #     plt.figure(2)
        #     for i in range(min(64, x.size(1))):
        #         plt.subplot(8,8,i+1)
        #         try:
        #             plt.imshow(x[0,i].detach().cpu().numpy())
        #         except:
        #             y = x6[0,i].detach().cpu().numpy()
        #             plt.imshow(y)
        #     plt.savefig('activations6.png')

        x, x7 = self.conv7(x)

        # if b_print:
        #     plt.figure(3)
        #     for i in range(min(64, x.size(1))):
        #         plt.subplot(8,8,i+1)
        #         try:
        #             plt.imshow(x[0,i].detach().cpu().numpy())
        #         except:
        #             y = x7[0,i].detach().cpu().numpy()
        #             plt.imshow(y)
        #     plt.savefig('activations7.png')

        x, x8 = self.conv8(x)

        # if b_print:
        #     plt.figure(4)
        #     for i in range(min(64, x.size(1))):
        #         plt.subplot(8,8,i+1)
        #         try:
        #             plt.imshow(x[0,i].detach().cpu().numpy())
        #         except:
        #             y = x8[0,i].detach().cpu().numpy()
        #             plt.imshow(y)
        #     plt.savefig('activations8.png')


        x2 = self.spatial_softmax(x)
        x = self.dropout(self.activation(self.conv_lin1(x.view(x.size(0), -1))))
        x = self.dropout(self.activation(self.conv_lin2(x)))

        x = torch.cat([x, x2], dim=1)#, tau

        #aux = self.aux(x)

        x = self.dropout(self.activation(self.fl1(torch.cat([low_dim, x], dim=1))))# , aux.detach()
        x = self.dropout(self.activation(self.fl2(x)))
        x = self.output(x)

        #'''
        x = torch.cat([x[:, :3], self.A_vec_to_quat(x[:, 3:-1]), x[:, -1:]], dim=1)
        '''
        norma = torch.tanh(x[:, 3:7])
        norm = norma*norma
        norm = norm.sum(dim=1, keepdim=True)
        norm = norm.sqrt()
        x = torch.cat([x[:, :3], norma/norm, torch.sigmoid(x[:, 7:])], dim=1)
        '''

        return x, None, None
