import torch

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from models.mlp import MLP

import numpy as np
import math 

import torch_scatter

'''
    等变卷积
    后续我们需要把它改为参数模糊
'''
class EquivSetConv(nn.Module):
    def __init__(self, in_features, out_features, mlp1_layers=1, mlp2_layers=1,
        mlp3_layers=1, aggr='add', alpha=0.5, dropout=0., normalization='None', input_norm=False):
        super().__init__()

        if mlp1_layers > 0:
            print('mlp1_layers', mlp1_layers)
            self.W1 = MLP(in_features, out_features, out_features, mlp1_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)   # input_norm: True
        else:
            self.W1 = nn.Identity()

        if mlp2_layers > 0:
            print('mlp2_layers', mlp2_layers)
            self.W2 = MLP(in_features+out_features, out_features, out_features, mlp2_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W2 = lambda X: X[..., in_features:]

        if mlp3_layers > 0:
            print('mlp3_layers', mlp3_layers)
            self.W = MLP(out_features, out_features, out_features, mlp3_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W = nn.Identity()
        self.aggr = aggr
        self.alpha = alpha
        self.dropout = dropout

    def reset_parameters(self):
        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W, MLP):
            self.W.reset_parameters()

    def forward(self, X, vertex, edges, X0):
        N = X.shape[-2]

        Xve = self.W1(X)[..., vertex, :] # [nnz, C] 切片操作，按照vertex中节点的index取出节点对应的特征
        Xe = torch_scatter.scatter(Xve, edges, dim=-2, reduce=self.aggr) # [E, C], reduce is 'mean' here as default 这里就直接按照edge进行取值，然后取平均值计算出超边特征
        
        Xev = Xe[..., edges, :] # [nnz, C] 切片操作，按照edges中超边的index取出超边对应的特征
        Xev = self.W2(torch.cat([X[..., vertex, :], Xev], -1))  # 合并超边特征和节点特征
        Xv = torch_scatter.scatter(Xev, vertex, dim=-2, reduce=self.aggr, dim_size=N) # [N, C]  按节点聚合超边特征，得到最后的节点特征

        X = Xv

        X = (1-self.alpha) * X + self.alpha * X0
        X = self.W(X)

        return X

class FuzzyEquivSetConv(nn.Module):
    def __init__(self, in_features, out_features, mlp1_layers=1, mlp2_layers=1,
                 mlp3_layers=1, aggr='add', alpha=0.5, dropout=0., normalization='None', input_norm=False):
        super().__init__()

        if mlp1_layers > 0:
            self.W1 = MLP(in_features, out_features, out_features, mlp1_layers,
                          dropout=dropout, Normalization=normalization, InputNorm=input_norm)  # input_norm: True
        else:
            self.W1 = nn.Identity()

        if mlp2_layers > 0:
            self.W2 = MLP(in_features + out_features, out_features, out_features, mlp2_layers,
                          dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W2 = lambda X: X[..., in_features:]

        if mlp3_layers > 0:
            self.W = MLP(out_features, out_features, out_features, mlp3_layers,
                         dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W = nn.Identity()
        self.aggr = aggr
        self.alpha = alpha
        self.dropout = dropout

    def reset_parameters(self):
        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W, MLP):
            self.W.reset_parameters()

    def forward(self, X, vertex, edges, X0):
        N = X.shape[-2]

        Xve = self.W1(X)[..., vertex, :]  # [nnz, C]
        Xe = torch_scatter.scatter(Xve, edges, dim=-2, reduce=self.aggr)  # [E, C], reduce is 'mean' here as default

        Xev = Xe[..., edges, :]  # [nnz, C]
        Xev = self.W2(torch.cat([X[..., vertex, :], Xev], -1))
        Xv = torch_scatter.scatter(Xev, vertex, dim=-2, reduce=self.aggr, dim_size=N)  # [N, C]

        X = Xv

        X = (1 - self.alpha) * X + self.alpha * X0
        X = self.W(X)

        return X

class JumpLinkConv(nn.Module):
    def __init__(self, in_features, out_features, mlp_layers=2, aggr='add', alpha=0.5):
        super().__init__()
        self.W = MLP(in_features, out_features, out_features, mlp_layers,
            dropout=0., Normalization='None', InputNorm=False)

        self.aggr = aggr
        self.alpha = alpha

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, X, vertex, edges, X0, beta=1.):
        N = X.shape[-2]

        Xve = X[..., vertex, :] # [nnz, C]
        Xe = torch_scatter.scatter(Xve, edges, dim=-2, reduce=self.aggr) # [E, C], reduce is 'mean' here as default
        
        Xev = Xe[..., edges, :] # [nnz, C]
        Xv = torch_scatter.scatter(Xev, vertex, dim=-2, reduce=self.aggr, dim_size=N) # [N, C]

        X = Xv

        Xi = (1-self.alpha) * X + self.alpha * X0
        X = (1-beta) * Xi + beta * self.W(Xi)

        return X

class MeanDegConv(nn.Module):
    def __init__(self, in_features, out_features, init_features=None, 
        mlp1_layers=1, mlp2_layers=1, mlp3_layers=2):
        super().__init__()
        if init_features is None:
            init_features = out_features
        self.W1 = MLP(in_features, out_features, out_features, mlp1_layers,
            dropout=0., Normalization='None', InputNorm=False)
        self.W2 = MLP(in_features+out_features+1, out_features, out_features, mlp2_layers,
            dropout=0., Normalization='None', InputNorm=False)
        self.W3 = MLP(in_features+out_features+init_features+1, out_features, out_features, mlp3_layers,
            dropout=0., Normalization='None', InputNorm=False)

    def reset_parameters(self):
        self.W1.reset_parameters()
        self.W2.reset_parameters()
        self.W3.reset_parameters()

    def forward(self, X, vertex, edges, X0):
        N = X.shape[-2]

        Xve = self.W1(X[..., vertex, :]) # [nnz, C]
        Xe = torch_scatter.scatter(Xve, edges, dim=-2, reduce='mean') # [E, C], reduce is 'mean' here as default

        deg_e = torch_scatter.scatter(torch.ones(Xve.shape[0], device=Xve.device), edges, dim=-2, reduce='sum')
        Xe = torch.cat([Xe, torch.log(deg_e)[..., None]], -1)

        Xev = Xe[..., edges, :] # [nnz, C]
        Xev = self.W2(torch.cat([X[..., vertex, :], Xev], -1))
        Xv = torch_scatter.scatter(Xev, vertex, dim=-2, reduce='mean', dim_size=N) # [N, C]

        deg_v = torch_scatter.scatter(torch.ones(Xev.shape[0], device=Xev.device), vertex, dim=-2, reduce='sum')
        X = self.W3(torch.cat([Xv, X, X0, torch.log(deg_v)[..., None]], -1))

        return X

'''
    EDGNN框架
'''
class EquivSetGNN(nn.Module):
    def __init__(self, num_features, num_classes, args):
        """UniGNNII

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        nhid = args.MLP_hidden
        act = {'Id': nn.Identity(), 'relu': nn.ReLU(), 'prelu':nn.PReLU()}
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_dropout) # 0.6 is chosen as default
        self.dropout = nn.Dropout(args.dropout) # 0.2 is chosen for GCNII

        self.in_channels = num_features
        self.hidden_channels = args.MLP_hidden
        self.output_channels = num_classes

        self.mlp1_layers = args.MLP_num_layers
        self.mlp2_layers = args.MLP_num_layers if args.MLP2_num_layers < 0 else args.MLP2_num_layers
        self.mlp3_layers = args.MLP_num_layers if args.MLP3_num_layers < 0 else args.MLP3_num_layers
        self.nlayer = args.All_num_layers
        self.edconv_type = args.edconv_type

        self.lin_in = torch.nn.Linear(num_features, args.MLP_hidden)
        if args.edconv_type == 'EquivSet':
            # 这里使用的是这个等变卷积核
            # mlp123_layers都有值
            # aggr: mean
            # normalization: ln
            # AllSet_input_norm: True
            self.conv = EquivSetConv(args.MLP_hidden, args.MLP_hidden, mlp1_layers=self.mlp1_layers, mlp2_layers=self.mlp2_layers,
                mlp3_layers=self.mlp3_layers, alpha=args.restart_alpha, aggr=args.aggregate,
                dropout=args.dropout, normalization=args.normalization, input_norm=args.AllSet_input_norm)
        elif args.edconv_type == 'JumpLink':
            self.conv = JumpLinkConv(args.MLP_hidden, args.MLP_hidden, mlp_layers=self.mlp1_layers, alpha=args.restart_alpha, aggr=args.aggregate)
        elif args.edconv_type == 'MeanDeg':
            self.conv = MeanDegConv(args.MLP_hidden, args.MLP_hidden, init_features=args.MLP_hidden, mlp1_layers=self.mlp1_layers,
                mlp2_layers=self.mlp2_layers, mlp3_layers=self.mlp3_layers)
        else:
            raise ValueError(f'Unsupported EDConv type: {args.edconv_type}')

        self.classifier = MLP(in_channels=args.MLP_hidden,
            hidden_channels=args.Classifier_hidden,
            out_channels=num_classes,
            num_layers=args.Classifier_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=False)

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        self.conv.reset_parameters()
        
        self.classifier.reset_parameters()

    def forward(self, data):
        x = data.x
        V, E = data.edge_index[0], data.edge_index[1]  # V表示节点index, E表示超边index (这里用的还是pyg的那一套表示方法，也可以看作超图的关联矩阵)
        lamda, alpha = 0.5, 0.1
        x = self.dropout(x)           # 先稀疏化
        x = F.relu(self.lin_in(x))    # 再降维 (可能为了后续方便处理)
        x0 = x
        for i in range(self.nlayer):
            x = self.dropout(x)
            x = self.conv(x, V, E, x0)
            x = self.act(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

class EquivDiffusion(nn.Module):
    def __init__(self, num_features, num_classes, args):

        super().__init__()

        mlp1_layers = args.MLP_num_layers
        mlp2_layers = args.MLP_num_layers if args.MLP2_num_layers < 0 else args.MLP2_num_layers

        self.W1 = MLP(num_features, args.MLP_hidden, args.MLP_hidden, mlp1_layers,
            dropout=args.dropout, Normalization=args.normalization, InputNorm=False)
        self.W2 = MLP(num_features+args.MLP_hidden, args.MLP_hidden, num_classes, mlp2_layers,
            dropout=args.dropout, Normalization=args.normalization, InputNorm=False)

        self.aggr = args.aggregate
        self.alpha = args.restart_alpha

    def reset_parameters(self):
        self.W1.reset_parameters()
        self.W2.reset_parameters()

    def forward(self, data):

        X = data.x
        vertex, edges = data.edge_index[0], data.edge_index[1]

        N = X.shape[-2]
        X0 = X

        Xve = self.W1(X[..., vertex, :]) # [nnz, C]
        Xe = torch_scatter.scatter(Xve, edges, dim=-2, reduce=self.aggr) # [E, C], reduce is 'mean' here as default
        
        Xev = Xe[..., edges, :] # [nnz, C]
        Xev = self.W2(torch.cat([X[..., vertex, :], Xev], -1))
        Xv = torch_scatter.scatter(Xev, vertex, dim=-2, reduce=self.aggr, dim_size=N) # [N, C]

        X = Xv

        X = (1-self.alpha) * X + self.alpha * X0

        return X