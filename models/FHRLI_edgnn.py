import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP
import torch_scatter

"Implemented by ED-HNN"
class EquivSetConv(nn.Module):
    def __init__(self, in_features, out_features, mlp1_layers=1, mlp2_layers=1,
                 mlp3_layers=1, aggr='add', alpha=0.5, dropout=0., normalization='None', input_norm=False):
        super().__init__()

        if mlp1_layers > 0:
            print('mlp1_layers', mlp1_layers)
            self.W1 = MLP(in_features, out_features, out_features, mlp1_layers,
                          dropout=dropout, Normalization=normalization, InputNorm=input_norm)  # input_norm: True
        else:
            self.W1 = nn.Identity()

        if mlp2_layers > 0:
            print('mlp2_layers', mlp2_layers)
            self.W2 = MLP(in_features + out_features, out_features, out_features, mlp2_layers,
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

        Xve = self.W1(X)[..., vertex, :]  # [nnz, C]
        Xe = torch_scatter.scatter(Xve, edges, dim=-2,
                                   reduce=self.aggr)  # [E, C]

        Xev = Xe[..., edges, :]  # [nnz, C]
        Xev = self.W2(torch.cat([X[..., vertex, :], Xev], -1))
        Xv = torch_scatter.scatter(Xev, vertex, dim=-2, reduce=self.aggr, dim_size=N)  # [N, C]

        X = Xv

        X = (1 - self.alpha) * X + self.alpha * X0
        X = self.W(X)

        return X

class FuzzyMLP(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(FuzzyMLP, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.InputNorm = InputNorm  # True

        self.w_b = nn.Parameter(torch.FloatTensor(size=(self.in_channels, self.out_channels)), requires_grad=True)
        self.w_a = nn.Parameter(torch.FloatTensor(size=(self.in_channels, self.out_channels)), requires_grad=True)
        self.w_c = nn.Parameter(torch.FloatTensor(size=(self.in_channels, self.out_channels)), requires_grad=True)

        self.b_b = nn.Parameter(torch.FloatTensor(size=(1, self.out_channels)), requires_grad=True)
        self.b_a = nn.Parameter(torch.FloatTensor(size=(1, self.out_channels)), requires_grad=True)
        self.b_c = nn.Parameter(torch.FloatTensor(size=(1, self.out_channels)), requires_grad=True)

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if InputNorm:
                self.normalization = nn.BatchNorm1d(in_channels)
            else:
                self.normalization = nn.Identity()

        elif Normalization == 'ln':
            if InputNorm:
                self.normalization = nn.LayerNorm(in_channels)
            else:
                self.normalization = nn.Identity()
        else:
            self.normalization = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.uniform_(self.w_b, 0.02, 0.05)
        nn.init.uniform_(self.w_a, 0.02, 0.05)
        nn.init.uniform_(self.w_c, 0.02, 0.05)
        nn.init.uniform_(self.b_b, 0.02, 0.05)
        nn.init.uniform_(self.b_a, 0.02, 0.05)
        nn.init.uniform_(self.b_c, 0.02, 0.05)

        if not (self.normalization.__class__.__name__ == 'Identity'):
            self.normalization.reset_parameters()

    def fuzziness_weight_decay_loss(self):
        param = [self.w_a, self.w_c, self.b_a, self.b_c]
        w_loss = 0.
        for p in param:
            w_loss += torch.sum(p * p)
        return 0.5 * w_loss

    def fuzziness_center_decay_loss(self):
        param = [self.w_b, self.b_b]
        w_loss = 0.
        for p in param:
            w_loss += torch.sum(p * p)
        return 0.5 * w_loss

    def interval_prod(self, interval_matrix1, interval_matrix2, bias=None):

        m1_l, m1_r = interval_matrix1
        m2_l, m2_r = interval_matrix2
        row = m1_l.shape[0]
        col = m2_l.shape[1]

        m1_l_train = m1_l
        m1_r_train = m1_r

        m1_l_reshape = torch.reshape(m1_l_train, (1, -1))
        m1_r_reshape = torch.reshape(m1_r_train, (1, -1))

        m2_l_reshape = torch.reshape(m2_l.unsqueeze(dim=0).expand(row, m2_l.shape[0], m2_l.shape[1]),
                                     (-1, col))
        m2_r_reshape = torch.reshape(m2_r.unsqueeze(dim=0).expand(row, m2_r.shape[0], m2_r.shape[1]),
                                     (-1, col))

        mt_l = m2_l_reshape.transpose(0, 1)
        mt_r = m2_r_reshape.transpose(0, 1)

        temp1 = m1_l_reshape * mt_l
        temp2 = m1_l_reshape * mt_r
        temp3 = m1_r_reshape * mt_l
        temp4 = m1_r_reshape * mt_r
        temp = torch.stack((temp1, temp2, temp3, temp4), dim=0)
        temp_min = torch.min(temp, dim=0)[0]
        temp_max = torch.max(temp, dim=0)[0]
        temp_min_t = temp_min.transpose(0, 1)
        temp_max_t = temp_max.transpose(0, 1)
        temp_min_reshape = torch.reshape(temp_min_t, (row, m2_l.shape[0], m2_l.shape[1]))
        temp_max_reshape = torch.reshape(temp_max_t, (row, m2_l.shape[0], m2_l.shape[1]))
        output_l = torch.sum(temp_min_reshape, dim=1)
        output_r = torch.sum(temp_max_reshape, dim=1)
        if bias is not None:
            b_l, b_r = bias
            output_l += b_l
            output_r += b_r

        return output_l, output_r

    def limit_parameters(self, param, lower=0., upper=1.):
        a, c = param
        a = F.relu(a)
        c = F.relu(c)
        return a, c

    def forward(self, x):
        hl, hr = x
        hl = self.normalization(hl)
        hr = self.normalization(hr)

        w_a, w_c = self.limit_parameters((self.w_a, self.w_c))
        b_a, b_c = self.limit_parameters((self.b_a, self.b_c))
        w_l = self.w_b - w_a
        w_r = self.w_b + w_c
        input_w = (w_l, w_r)
        b_l = self.b_b - b_a
        b_r = self.b_b + b_c
        input_b = (b_l, b_r)

        output_l, output_r = self.interval_prod((hl, hr), input_w, input_b)
        output = (output_l, output_r)

        return output


class CrispToFuzzyConv(nn.Module):

    def __init__(self, in_features, out_features, aggr='add', alpha=0.5, dropout=0., bias=True, Normalization='ln', InputNorm=False):
        super().__init__()
        self.bias = bias

        self.w_b = nn.Parameter(torch.FloatTensor(size=(2 * in_features, out_features)), requires_grad=True)
        self.w_a = nn.Parameter(torch.FloatTensor(size=(2 * in_features, out_features)), requires_grad=True)
        self.w_c = nn.Parameter(torch.FloatTensor(size=(2 * in_features, out_features)), requires_grad=True)

        if self.bias is True:
            self.b_b = nn.Parameter(torch.FloatTensor(size=(1, out_features)), requires_grad=True)
            self.b_a = nn.Parameter(torch.FloatTensor(size=(1, out_features)), requires_grad=True)
            self.b_c = nn.Parameter(torch.FloatTensor(size=(1, out_features)), requires_grad=True)

        self.aggr = aggr
        self.alpha = alpha
        self.dropout = dropout

        self.W2 = lambda X: X[..., in_features:]

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if InputNorm:
                self.normalization = nn.BatchNorm1d(2 * in_features)
            else:
                self.normalization = nn.Identity()

        elif Normalization == 'ln':
            if InputNorm:
                self.normalization = nn.LayerNorm(2 * in_features)
            else:
                self.normalization = nn.Identity()
        else:
            self.normalization = nn.Identity()

    def reset_parameters(self):
        nn.init.uniform_(self.w_b, 0.02, 0.05)
        nn.init.uniform_(self.w_a, 0.02, 0.05)
        nn.init.uniform_(self.w_c, 0.02, 0.05)
        nn.init.uniform_(self.b_b, 0.02, 0.05)
        nn.init.uniform_(self.b_a, 0.02, 0.05)
        nn.init.uniform_(self.b_c, 0.02, 0.05)

        if not (self.normalization.__class__.__name__ == 'Identity'):
            self.normalization.reset_parameters()

    def fuzziness_center_decay_loss(self):
        param = [self.w_b, self.b_b]
        w_loss = 0.
        for p in param:
            w_loss += torch.sum(p * p)
        return 0.5 * w_loss

    def fuzziness_weight_decay_loss(self):
        param = [self.w_a, self.w_c, self.b_a, self.b_c]
        w_loss = 0.
        for p in param:
            w_loss += torch.sum(p * p)
        return 0.5 * w_loss

    def fuzzy_scope(self, input, scope):
        return torch.mm(torch.abs(input), scope)

    def forward(self, X, vertex, edges, X0):
        N = X.shape[-2]

        Xve = X[..., vertex, :]  # [nnz, C]
        Xe = torch_scatter.scatter(Xve, edges, dim=-2, reduce=self.aggr)  # [E, C], reduce is 'mean' here as default

        Xev = Xe[..., edges, :]  # [nnz, C]
        Xev = torch.cat([X[..., vertex, :], Xev], -1)
        Xv = torch_scatter.scatter(Xev, vertex, dim=-2, reduce=self.aggr, dim_size=N)  # [N, C]
        X = Xv
        X = self.normalization(X)
        center = torch.mm(X, self.w_b) + self.b_b
        scope_l = self.fuzzy_scope(X, self.w_a)
        scope_r = self.fuzzy_scope(X, self.w_c)
        scope_l += self.b_a
        scope_r += self.b_c

        HL = center - scope_l
        HR = center + scope_r

        return center, HL, HR


class Feature_cut(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

    def forward(self, X):
        HL, HR = X
        hl = HL[..., self.in_features:]
        hr = HR[..., self.in_features:]
        return hl, hr


class FuzzyEquivSetConv(nn.Module):
    def __init__(self, in_features, out_features, mlp1_layers=1, mlp2_layers=1,
                 mlp3_layers=1, aggr='add', alpha=0.5, dropout=0., normalization='None', input_norm=False):
        super().__init__()

        if mlp1_layers > 0:
            self.W1 = FuzzyMLP(in_features, out_features, out_features, mlp1_layers,
                               dropout=dropout, Normalization=normalization, InputNorm=input_norm)  # input_norm: True
        else:
            self.W1 = nn.Identity()

        if mlp2_layers > 0:
            self.W2 = FuzzyMLP(in_features + out_features, out_features, out_features, mlp2_layers,
                               dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W2 = Feature_cut(in_features)

        if mlp3_layers > 0:
            self.W = FuzzyMLP(in_features * 2, out_features, out_features, mlp3_layers,
                              dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W = nn.Identity()
        self.aggr = aggr
        self.alpha = alpha
        self.dropout = dropout

    def reset_parameters(self):
        if isinstance(self.W1, FuzzyMLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, FuzzyMLP):
            self.W2.reset_parameters()
        if isinstance(self.W, FuzzyMLP):
            self.W.reset_parameters()

    def fuzziness_weight_decay_loss(self):
        w_loss = 0.
        n_layers = 0
        if isinstance(self.W1, FuzzyMLP):
            w_loss += self.W1.fuzziness_weight_decay_loss()
            n_layers += 1
        if isinstance(self.W2, FuzzyMLP):
            w_loss += self.W2.fuzziness_weight_decay_loss()
            n_layers += 1
        if isinstance(self.W, FuzzyMLP):
            w_loss += self.W.fuzziness_weight_decay_loss()
            n_layers += 1
        return w_loss

    def fuzziness_center_decay_loss(self):
        w_loss = 0.
        n_layers = 0
        if isinstance(self.W1, FuzzyMLP):
            w_loss += self.W1.fuzziness_center_decay_loss()
            n_layers += 1
        if isinstance(self.W2, FuzzyMLP):
            w_loss += self.W2.fuzziness_center_decay_loss()
            n_layers += 1
        if isinstance(self.W, FuzzyMLP):
            w_loss += self.W.fuzziness_center_decay_loss()
            n_layers += 1
        return w_loss

    def forward(self, x, vertex, edges, x0):
        hl, hr = x
        hl0, hr0 = x0
        N = hl.shape[-2]

        Xve_l, Xve_r = self.W1(x)
        Xve_hl = Xve_l[..., vertex, :]
        Xve_hr = Xve_r[..., vertex, :]

        Xe_hl = torch_scatter.scatter(Xve_hl, edges, dim=-2, reduce=self.aggr)
        Xe_hr = torch_scatter.scatter(Xve_hr, edges, dim=-2, reduce=self.aggr)

        Xev_l = Xe_hl[..., edges, :]  # [nnz, C]
        Xev_r = Xe_hr[..., edges, :]
        Xev_hl, Xev_hr = torch.cat([hl[..., vertex, :], Xev_l], -1), torch.cat([hr[..., vertex, :], Xev_r], -1)
        Xv_hl = torch_scatter.scatter(Xev_hl, vertex, dim=-2, reduce=self.aggr, dim_size=N)  # [N, C]
        Xv_hr = torch_scatter.scatter(Xev_hr, vertex, dim=-2, reduce=self.aggr, dim_size=N)  # [N, C]
        X_hl = Xv_hl
        X_hr = Xv_hr

        X = self.W((X_hl, X_hr))

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

        Xve = X[..., vertex, :]  # [nnz, C]
        Xe = torch_scatter.scatter(Xve, edges, dim=-2, reduce=self.aggr)  # [E, C], reduce is 'mean' here as default

        Xev = Xe[..., edges, :]  # [nnz, C]
        Xv = torch_scatter.scatter(Xev, vertex, dim=-2, reduce=self.aggr, dim_size=N)  # [N, C]

        X = Xv

        Xi = (1 - self.alpha) * X + self.alpha * X0
        X = (1 - beta) * Xi + beta * self.W(Xi)

        return X


class MeanDegConv(nn.Module):
    def __init__(self, in_features, out_features, init_features=None,
                 mlp1_layers=1, mlp2_layers=1, mlp3_layers=2):
        super().__init__()
        if init_features is None:
            init_features = out_features
        self.W1 = MLP(in_features, out_features, out_features, mlp1_layers,
                      dropout=0., Normalization='None', InputNorm=False)
        self.W2 = MLP(in_features + out_features + 1, out_features, out_features, mlp2_layers,
                      dropout=0., Normalization='None', InputNorm=False)
        self.W3 = MLP(in_features + out_features + init_features + 1, out_features, out_features, mlp3_layers,
                      dropout=0., Normalization='None', InputNorm=False)

    def reset_parameters(self):
        self.W1.reset_parameters()
        self.W2.reset_parameters()
        self.W3.reset_parameters()

    def forward(self, X, vertex, edges, X0):
        N = X.shape[-2]

        Xve = self.W1(X[..., vertex, :])  # [nnz, C]
        Xe = torch_scatter.scatter(Xve, edges, dim=-2, reduce='mean')  # [E, C], reduce is 'mean' here as default

        deg_e = torch_scatter.scatter(torch.ones(Xve.shape[0], device=Xve.device), edges, dim=-2, reduce='sum')
        Xe = torch.cat([Xe, torch.log(deg_e)[..., None]], -1)

        Xev = Xe[..., edges, :]  # [nnz, C]
        Xev = self.W2(torch.cat([X[..., vertex, :], Xev], -1))
        Xv = torch_scatter.scatter(Xev, vertex, dim=-2, reduce='mean', dim_size=N)  # [N, C]

        deg_v = torch_scatter.scatter(torch.ones(Xev.shape[0], device=Xev.device), vertex, dim=-2, reduce='sum')
        X = self.W3(torch.cat([Xv, X, X0, torch.log(deg_v)[..., None]], -1))

        return X


class TSK_input_fuzzifier(nn.Module):
    def __init__(self, in_dim, n_rules, output_dim1):
        super(TSK_input_fuzzifier, self).__init__()
        self.in_dim = in_dim
        self.n_rules = n_rules
        self.output_dim1 = output_dim1
        self.eps = 1e-15

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.build_model()
        self.to(self.device)

    def build_model(self):
        self.Cs = nn.Parameter(torch.FloatTensor(size=(self.in_dim, self.n_rules)), requires_grad=True)
        self.Vs = nn.Parameter(torch.FloatTensor(size=self.Cs.size()), requires_grad=True)

    def reset_parameters(self):
        # nn.init.normal_(self.Cs, mean=1, std=0.5)
        # nn.init.normal_(self.Vs, mean=1, std=0.5)
        nn.init.normal_(self.Cs, mean=1, std=0.2)
        nn.init.normal_(self.Vs, mean=1, std=0.2)
        # torch.nn.init.xavier_normal_(self.Cs)
        # torch.nn.init.normal_(self.Vs, mean=1, std=0.2)

    def fuzzify(self, features):
        fz_degree = -(features.unsqueeze(dim=2) - self.Cs) ** 2 / ((2 * self.Vs ** 2) + self.eps)
        fz_degree = torch.exp(fz_degree)
        weighted_fz_degree = torch.max(fz_degree, dim=2)[0]

        fz_features = features * weighted_fz_degree

        return fz_degree, fz_features

class FHRLI_EquivSetGNN(nn.Module):
    def __init__(self, num_features, num_classes, args):
        super().__init__()
        nhid = args.MLP_hidden
        act = {'Id': nn.Identity(), 'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_dropout)
        self.dropout = nn.Dropout(args.dropout)

        self.in_channels = num_features
        self.hidden_channels = args.MLP_hidden
        self.output_channels = num_classes
        self.n_rules = args.n_rules

        self.mlp1_layers = args.MLP_num_layers
        self.mlp2_layers = args.MLP_num_layers if args.MLP2_num_layers < 0 else args.MLP2_num_layers
        self.mlp3_layers = args.MLP_num_layers if args.MLP3_num_layers < 0 else args.MLP3_num_layers
        self.nlayer = args.All_num_layers
        self.edconv_type = args.edconv_type

        self.input_fuzzifier = TSK_input_fuzzifier(num_features, self.n_rules, self.hidden_channels)
        self.lin_in = torch.nn.Linear(num_features, args.MLP_hidden)
        if args.edconv_type == 'EquivSet':
            self.conv = EquivSetConv(args.MLP_hidden, args.MLP_hidden, mlp1_layers=self.mlp1_layers,
                                     mlp2_layers=self.mlp2_layers,
                                     mlp3_layers=self.mlp3_layers, alpha=args.restart_alpha, aggr=args.aggregate,
                                     dropout=args.dropout, normalization=args.normalization,
                                     input_norm=args.AllSet_input_norm)
        elif args.edconv_type == 'JumpLink':
            self.conv = JumpLinkConv(args.MLP_hidden, args.MLP_hidden, mlp_layers=self.mlp1_layers,
                                     alpha=args.restart_alpha, aggr=args.aggregate)
        elif args.edconv_type == 'MeanDeg':
            self.conv = MeanDegConv(args.MLP_hidden, args.MLP_hidden, init_features=args.MLP_hidden,
                                    mlp1_layers=self.mlp1_layers,
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
        self.input_fuzzifier.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data):
        x = data.x
        V, E = data.edge_index[0], data.edge_index[1]
        _, x = self.input_fuzzifier.fuzzify(x)
        x = self.dropout(x)
        x = F.relu(self.lin_in(x))
        x0 = x
        for i in range(self.nlayer):
            x = self.dropout(x)
            x = self.conv(x, V, E, x0)
            x = self.act(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class FHRLI_EquivSetGNN_PF(nn.Module):
    def __init__(self, num_features, num_classes, num_nodes, args):
        super().__init__()
        nhid = args.MLP_hidden
        act = {'Id': nn.Identity(), 'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_dropout)
        self.dropout = nn.Dropout(args.dropout)

        self.in_channels = num_features
        self.hidden_channels = args.MLP_hidden
        self.output_channels = num_classes
        self.n_rules = args.n_rules
        self.n_nodes = num_nodes

        self.mlp1_layers = args.MLP_num_layers
        self.mlp2_layers = args.MLP_num_layers if args.MLP2_num_layers < 0 else args.MLP2_num_layers
        self.mlp3_layers = args.MLP_num_layers if args.MLP3_num_layers < 0 else args.MLP3_num_layers
        self.nlayer = args.All_num_layers
        self.edconv_type = args.edconv_type

        self.input_fuzzifier = TSK_input_fuzzifier(num_features, self.n_rules, self.hidden_channels)
        self.fconv_1 = CrispToFuzzyConv(num_features, args.MLP_hidden, alpha=args.restart_alpha, aggr=args.aggregate,
                                        dropout=args.dropout, Normalization=args.normalization,
                                         InputNorm=args.AllSet_input_norm)
        self.fconv_2 = FuzzyEquivSetConv(args.MLP_hidden, num_classes, mlp1_layers=self.mlp1_layers,
                                         mlp2_layers=self.mlp2_layers,
                                         mlp3_layers=self.mlp3_layers, alpha=args.restart_alpha, aggr=args.aggregate,
                                         dropout=args.dropout, normalization=args.normalization,
                                         input_norm=args.AllSet_input_norm)
        self.center = nn.Parameter(torch.FloatTensor(size=(self.n_nodes, num_classes)), requires_grad=True)
        self.classifier = MLP(in_channels=args.MLP_hidden,
                              hidden_channels=args.Classifier_hidden,
                              out_channels=num_classes,
                              num_layers=args.Classifier_num_layers,
                              dropout=args.dropout,
                              Normalization=args.normalization,
                              InputNorm=False)

    def reset_parameters(self):
        self.fconv_1.reset_parameters()
        self.fconv_2.reset_parameters()
        self.input_fuzzifier.reset_parameters()
        self.classifier.reset_parameters()
        nn.init.normal_(self.center, mean=0.5, std=0.1)

    def forward(self, data):
        x = data.x
        V, E = data.edge_index[0], data.edge_index[1]
        _, x = self.input_fuzzifier.fuzzify(x)

        x = self.dropout(x)
        x0 = x
        center, hl, hr = self.fconv_1(x, V, E, x0)
        x0 = (hl, hr)
        for i in range(self.nlayer):
            hl, hr = self.dropout(hl), self.dropout(hr)
            x = (hl, hr)
            hl, hr = self.fconv_2(x, V, E, x0)

        hl, hr = self.dropout(hl), self.dropout(hr)
        defuzziness = self.update_center(hl, hr)
        return defuzziness

    def update_center(self, hl, hr):
        center = torch.clamp(self.center, min=0., max=1.)
        if len(hl.shape) <= 2:
            return hl + center * (hr - hl)
        else:
            return hl + torch.einsum('ij,ijk->ijk', center, hr - hl)

    def fuzziness_weight_decay_loss(self):
        input_fuzzifier_wd_loss = self.fconv_1.fuzziness_weight_decay_loss()
        fuzzy_conv_wd_loss = self.fconv_2.fuzziness_weight_decay_loss()

        return input_fuzzifier_wd_loss, fuzzy_conv_wd_loss




