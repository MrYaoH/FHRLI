import math
import numpy as np

import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter

class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(MLP, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm  # True

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ == 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

    def flops(self, x):
        num_samples = np.prod(x.shape[:-1])
        flops = num_samples * self.in_channels # first normalization
        flops += num_samples * self.in_channels * self.hidden_channels # first linear layer
        flops += num_samples * self.hidden_channels # first relu layer

        # flops for each layer
        per_layer = num_samples * self.hidden_channels * self.hidden_channels
        per_layer += num_samples * self.hidden_channels # relu + normalization
        flops += per_layer * (len(self.lins) - 2)

        flops += num_samples * self.out_channels * self.hidden_channels # last linear layer

        return flops


class PlainMLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(PlainMLP, self).__init__()
        self.lins = nn.ModuleList()

        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class TSK(nn.Module):
    def __init__(self, in_features, num_classes, n_rules=3):
        super(TSK, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.n_rules = n_rules

        self.build_model()

    def build_model(self):
        self.ampli = 0.
        self.eps = 1e-8

        self.Cons = torch.FloatTensor(size=(self.n_rules, self.in_features, self.num_classes))  # 这个应该是后续的线性投影部分
        self.Bias = torch.FloatTensor(size=(1, self.n_rules, self.num_classes))
        self.Cs = torch.FloatTensor(size=(self.in_features, self.n_rules))  # 保存初始化聚类中心，也就是模糊隶属度函数的均值
        self.Vs = torch.FloatTensor(size=self.Cs.size())  # 隶属度函数的方差

        self.Cons = nn.Parameter(self.Cons, requires_grad=True)  # 参数化，这些之后需要进行学习的
        self.Bias = nn.Parameter(self.Bias, requires_grad=True)
        self.Cs = nn.Parameter(self.Cs, requires_grad=True)
        self.Vs = nn.Parameter(self.Vs, requires_grad=True)

    def reset_parameters(self):
        # torch.nn.init.xavier_normal_(self.Cons)
        nn.init.uniform_(self.Cons, -1, 1)
        torch.nn.init.constant_(self.Bias, 0)
        torch.nn.init.xavier_normal_(self.Cs)
        torch.nn.init.normal_(self.Vs, mean=1, std=0.2)

        # nn.init.xavier_normal_(self.Cs)
        # nn.init.normal_(self.Vs, mean=1, std=0.2)
        # nn.init.uniform_(self.Cons, -1, 1)
        # nn.init.constant_(self.Bias, 0)


    def forward(self, fuzzy_input):
        sim = torch.sum(-(fuzzy_input.unsqueeze(dim=-1) - self.Cs) ** 2 / (2 * self.Vs ** 2), dim=1) # sum_n_walk, n_rules
        frs = F.softmax(sim, dim=-1)  # fire-level

        x_rep = fuzzy_input.unsqueeze(dim=1).expand([fuzzy_input.size(0), self.n_rules, fuzzy_input.size(1)])  # [sum_n_walk, rules, feat]
        cons1 = torch.einsum('ijk,jkl->ijl', [x_rep, self.Cons])  # Cons shape : [rules, feat, out]   ——> [sum_n_walk, rules, out]
        cons1 = cons1 + self.Bias   # fuzzy number predict y_i = f(x)

        out = torch.matmul(frs.unsqueeze(dim=1), cons1).squeeze(dim=1)   # defuzzy  [sum_n_walk, dim]

        return out

    def l2_loss(self):
        return torch.sum(self.Cons ** 2)
