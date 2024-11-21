import torch
from torch import Tensor, nn
from torch.nn import MSELoss, Linear
from torch.nn import functional as func

from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GATConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm, GCNConv

from utils.utils import BernoulliLoss, GMRFLoss, GMRFSamplingLoss
import numpy as np

class SGConv(MessagePassing):
    """
    Convolution layer for a simplified graph convolutional network.
    """

    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        """
        Class initializer.
        """
        kwargs.setdefault('aggr', 'add')
        super(SGConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.add_self_loops = add_self_loops

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters in the linear layer.
        """
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        """
        Run forward propagation.
        """
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype)

        x = self.lin(x)

        for k in range(self.K):
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        """
        Message function for the PyG implementation.
        """
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        """
        Message and aggregate function for the PyG implementation.
        """
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        """
        Make a representation of this layer.
        """
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)


# noinspection PyMethodOverriding
class SAGEConv(MessagePassing):
    """
    Convolution layer for GraphSAGE.
    """

    def __init__(self, in_channels, out_channels, normalize=False, bias=True, **kwargs):
        """
        Class initializer.
        """
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters in the layers.
        """
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):
        """
        Run forward propagation.
        """
        if isinstance(x, Tensor):
            x = (x, x)
        out = self.lin_l(x[0])
        out = self.propagate(edge_index, x=out, size=size)
        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)
        if self.normalize:
            out = func.normalize(out, p=2., dim=-1)
        return out

    def message(self, x_j):
        """
        Message function for the PyG implementation.
        """
        return x_j

    def message_and_aggregate(self, adj_t, x):
        """
        Message and aggregate function for the PyG implementation.
        """
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        """
        Make a representation of this layer.
        """
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

class SGC(nn.Module):
    """
    Implementation of a simplified graph convolutional network.
    """

    def __init__(self, num_features, num_classes, num_layers=2, bias=True):
        """
        Class initializer.
        """
        super().__init__()
        self.layer = SGConv(num_features, num_classes, K=num_layers, bias=bias)

    def forward(self, x, edges):
        """
        Run forward propagation.
        """
        return self.layer(x, edges)


class GNN(nn.Module):
    """
    Class that supports general graph neural networks.
    """
    def __init__(self, num_features, num_classes, num_layers=2, hidden_size=16,
                 dropout=0.5, residual=True, conv='gcn'):
        """
        Class initializer.
        """
        super().__init__()
        self.residual = residual
        self.conv = conv
        self.relu = nn.ReLU()
        self.drop_prob = dropout
        self.dropout = nn.Dropout()

        layers = []
        for i in range(num_layers):
            in_size = num_features if i == 0 else hidden_size
            out_size = num_classes if i == num_layers - 1 else hidden_size
            layers.append(self.to_conv(in_size, out_size))
        self.layers = nn.ModuleList(layers)

    def to_conv(self, in_size, out_size, heads=8):
        """
        Make a convolution layer based on the current type.
        """
        if self.conv == 'gcn':
            return GCNConv(in_size, out_size)
        elif self.conv == 'sage':
            return SAGEConv(in_size, out_size)
        elif self.conv == 'gat':
            return GATConv(in_size, out_size // heads, heads=heads, dropout=self.drop_prob)
        else:
            raise ValueError(self.conv)

    def forward(self, x, edges):
        """
        Run forward propagation.
        """
        out = x
        for i, layer in enumerate(self.layers[:-1]):
            out2 = out
            if i > 0:
                out2 = self.dropout(out2)
            out2 = layer(out2, edges)
            out2 = self.relu(out2)
            if i > 0 and self.residual:
                out2 = out2 + out
            out = out2
        out = self.dropout(out)
        return self.layers[-1](out, edges)


def to_x_loss(x_loss):
    """
    Make a loss term for estimating node features.
    """
    if x_loss in ['base', 'balanced']:
        return BernoulliLoss(x_loss)
    elif x_loss == 'gaussian':
        return MSELoss()
    else:
        raise ValueError(x_loss)


class Features(nn.Module):
    """
    Class that supports various types of node features.
    """

    def __init__(self, edge_index, num_nodes, version, obs_nodes=None, obs_features=None,
                 dropout=0):
        """
        Class initializer.
        """
        super().__init__()
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.indices = None
        self.values = None
        self.shape = None

        if version == 'diag':
            indices, values, shape = self.to_diag_features()
        elif version == 'degree':
            indices, values, shape = self.to_degree_features(edge_index)
        elif version == 'diag-degree':
            indices, values, shape = self.to_diag_degree_features(edge_index)
        elif version == 'obs-diag':
            indices, values, shape = self.to_obs_diag_features(obs_nodes, obs_features)
        else:
            raise ValueError(version)

        self.indices = nn.Parameter(indices, requires_grad=False)
        self.values = nn.Parameter(values, requires_grad=False)
        self.shape = shape

    def forward(self):
        """
        Make a feature Tensor from the current information.
        """
        return torch.sparse_coo_tensor(self.indices, self.values, size=self.shape,
                                       device=self.indices.device)

    def to_diag_features(self):
        """
        Make a diagonal feature matrix.
        """
        nodes = torch.arange(self.num_nodes)
        if self.training and self.dropout > 0:
            nodes = nodes[torch.rand(self.num_nodes) > self.dropout]
        indices = nodes.view(1, -1).expand(2, -1).contiguous()
        values = torch.ones(self.num_nodes)
        shape = self.num_nodes, self.num_nodes
        return indices, values, shape

    def to_degree_features(self, edge_index):
        """
        Make a degree-based feature matrix.
        """
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                             sparse_sizes=(self.num_nodes, self.num_nodes))
        degree = adj_t.sum(dim=0).long()
        degree_list = torch.unique(degree)
        degree_map = torch.zeros_like(degree)
        degree_map[degree_list] = torch.arange(len(degree_list))
        indices = torch.stack([torch.arange(self.num_nodes), degree_map[degree]], dim=0)
        values = torch.ones(indices.size(1))
        shape = self.num_nodes, indices[1, :].max() + 1
        return indices, values, shape

    def to_diag_degree_features(self, edge_index):
        """
        Combine the diagonal and degree-based feature matrices.
        """
        indices1, values1, shape1 = self.to_diag_features()
        indices2, values2, shape2 = self.to_degree_features(edge_index)
        indices = torch.cat([indices1, indices2], dim=1)
        values = torch.cat([values1, values2])
        shape = shape1[0], shape1[1] + shape2[1]
        return indices, values, shape

    def to_obs_diag_features(self, obs_nodes, obs_features):
        """
        Combine the observed features and diagonal ones.
        """
        num_features = obs_features.size(1) + self.num_nodes - len(obs_nodes)
        row, col = torch.nonzero(obs_features, as_tuple=True)
        indices1 = torch.stack([obs_nodes[row], col])
        values1 = obs_features[row, col]

        nodes2 = torch.arange(self.num_nodes)
        nodes2[obs_nodes] = False
        nodes2 = torch.nonzero(nodes2).view(-1)
        indices2 = torch.stack([nodes2, torch.arange(len(nodes2))])
        indices2[1, :] += obs_features.size(1)
        values2 = torch.ones(indices2.size(1))

        indices = torch.cat([indices1, indices2], dim=1)
        values = torch.cat([values1, values2], dim=0)
        shape = self.num_nodes, num_features
        return indices, values, shape


class Encoder(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, dropout, conv):
        super().__init__()
        if conv == 'sgc':
            self.model = SGC(num_features, hidden_size, num_layers)
        elif conv == 'lin':
            self.model = nn.Linear(num_features, hidden_size)
        elif conv in ['gcn', 'sage', 'gat']:
            self.model = GNN(num_features, hidden_size, num_layers, hidden_size, dropout, conv=conv)
        else:
            raise ValueError()

    def forward(self, features, edge_index):
        if isinstance(self.model, nn.Linear):
            return self.model(features)
        else:
            return self.model(features, edge_index)


class Decoder(nn.Module):
    """
    Encoder network in the proposed framework.
    """

    def __init__(self, input_size, output_size, hidden_size=16, num_layers=2, dropout=0.5):
        """
        Class initializer.
        """
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            out_size = output_size if i == num_layers - 1 else hidden_size
            if i > 0:
                layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            layers.append(nn.Linear(in_size, out_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Run forward propagation.
        """
        return self.layers(x)


class Identity(nn.Module):
    """
    PyTorch module that implements the identity function.
    """

    def __init__(self):
        """
        Class initializer.
        """
        super().__init__()

    def forward(self, x):
        """
        Run forward propagation.
        """
        return x


class UnitNorm(nn.Module):
    """
    Unit normalization of latent variables.
    """

    def __init__(self):
        """
        Class initializer.
        """
        super().__init__()

    def forward(self, vectors):
        """
        Run forward propagation.
        """
        valid_index = (vectors != 0).sum(1, keepdims=True) > 0
        vectors = torch.where(valid_index, vectors, torch.randn_like(vectors))
        return vectors / (vectors ** 2).sum(1, keepdims=True).sqrt()


class EmbNorm(nn.Module):
    """
    The normalization of node representations.
    """

    def __init__(self, hidden_size, function='unit', affine=True):
        """
        Class initializer.
        """
        super().__init__()
        if function == 'none':
            self.norm = Identity()
        elif function == 'unit':
            self.norm = UnitNorm()
        elif function == 'batchnorm':
            self.norm = nn.BatchNorm1d(hidden_size, affine=affine)
        elif function == 'layernorm':
            self.norm = nn.LayerNorm(hidden_size, elementwise_affine=affine)
        else:
            raise ValueError(function)

    def forward(self, vectors):
        """
        Run forward propagation.
        """
        return self.norm(vectors)

class SVGA(nn.Module):
    """
    Class of our proposed method.
    """

    def __init__(self, edge_index, graphs, num_nodes, num_features, num_classes, hidden_size=256, lamda=1,
                 beta=0.1, num_layers=2, conv='gcn', dropout=0.5, x_type='diag', x_loss='balanced',
                 emb_norm='unit', obs_nodes=None, obs_features=None, dec_bias=False, is_continuous=True):
        """
        Class initializer.
        """
        super().__init__()
        self.lamda = lamda
        self.dropout = nn.Dropout(dropout)

        self.features = Features(edge_index, num_nodes, x_type, obs_nodes, obs_features)
        self.encoder = Encoder(self.features.shape[1], hidden_size, num_layers, dropout, conv)
        self.emb_norm = EmbNorm(hidden_size, emb_norm)

        self.x_decoder = nn.Linear(hidden_size, num_features, bias=dec_bias)
        self.y_decoder = nn.Linear(hidden_size, 1, bias=dec_bias)  if is_continuous else nn.Linear(hidden_size, num_classes, bias=dec_bias)

        self.x_loss = to_x_loss(x_loss)
        self.y_loss = MSELoss() if is_continuous else nn.CrossEntropyLoss()
        self.kld_loss = GMRFLoss(beta)

    def forward(self, edge_index, data, graphs,for_loss=False):
        """
        Run forward propagation.
        """
        # z_TF_GP = self.emb_norm(self.encoder(self.features()[data.TF_mask | data.GP_mask], graphs['TF', 'TF_GP', 'GP']))
        z = self.emb_norm(self.encoder(self.features(), edge_index))
        z_dropped = self.dropout(z)
        x_hat = self.x_decoder(z_dropped)
        y_hat = self.y_decoder(z_dropped)
        if for_loss:
            return z, x_hat, y_hat
        return x_hat, y_hat

    def to_y_loss(self, y_hat, y_nodes, y_labels,is_continuous=True):
        """
        Make a loss term for observed labels.
        """
        if y_nodes is not None and y_labels is not None:
            if is_continuous:
                return self.y_loss(y_hat.float(),y_labels.float())
            else:
                return self.y_loss(y_hat.type(torch.long), y_labels.type(torch.long))
        else:
            return torch.zeros(1, device=y_hat.device)

    def to_kld_loss(self, z, edge_index):
        """
        Make a KL divergence regularizer.
        """
        if self.lamda > 0:
            return self.lamda * self.kld_loss(z, edge_index)
        else:
            return torch.zeros(1, device=z.device)

    def to_losses(self, edge_index, data,graphs, x_features, is_continuous,y_nodes=None, y_labels=None):
        """
        Make three loss terms for the training.
        """
        z, x_hat, y_hat = self.forward(edge_index, data, graphs,for_loss=True)
        l1 = self.x_loss(x_hat[data.train_mask].float(), data.x[data.train_mask].float()) # 计算训练集中节点feat和预测feat之间的损失
        l2 = self.to_y_loss(y_hat[data.train_mask], y_nodes, torch.reshape(y_labels,(-1, 1)),is_continuous) # 计算训练集中节点标签和预测标签之间的损失
        l3 = self.to_kld_loss(z, edge_index)
        return l1, l2,  l3

@torch.no_grad()
def to_f1_score(input, target, epsilon=1e-8):
    """
    Compute the F1 score from a prediction.
    """
    assert (target < 0).int().sum() == 0
    tp = ((input > 0) & (target > 0)).sum()
    fp = ((input > 0) & (target == 0)).sum()
    fn = ((input <= 0) & (target > 0)).sum()
    return (tp / (tp + (fp + fn) / 2 + epsilon)).item()


@torch.no_grad()
def to_recall(input, target, k=20):
    """
    Compute the recall score from a prediction.
    """
    pred = input.topk(k, dim=1, sorted=False)[1]
    row_index = torch.arange(target.size(0))
    target_list = []
    for i in range(k):
        target_list.append(target[row_index, pred[:, i]])
    num_pred = torch.stack(target_list, dim=1).sum(dim=1)
    num_true = target.sum(dim=1)
    return (num_pred[num_true > 0] / num_true[num_true > 0]).mean().item()

@torch.no_grad()
def to_r2(input, target):
    """
    Compute the CORR (or the R square) score from a prediction.
    """
    a = ((input - target) ** 2).sum()
    b = ((target - target.mean(dim=0)) ** 2).sum()
    return (1 - a / b).item()


@torch.no_grad()
def to_rmse(input, target):
    """
    Compute the RMSE score from a prediction.
    """
    return ((input - target) ** 2).mean(dim=1).sqrt().mean().item()

def update_model(step,model,edge_index,data,graphs,x_features,optimizer, y_nodes,y_labels,is_continuous):
    model.train()
    losses = model.to_losses(edge_index, data, graphs,x_features,is_continuous,y_nodes,y_labels)
    if step:
        optimizer.zero_grad()
        sum(losses).backward()
        optimizer.step()
    return tuple(l.item() for l in losses)

@torch.no_grad()
def evaluate_model(model,edge_index,x_nodes,val_nodes,is_continuous,x_all, data, graphs):
    model.eval()
    x_hat_, _ = model(edge_index, data, graphs)
    out_list = []
    k = 20 if x_all.shape[1] >= 20 else x_all.shape[1]
    for nodes in [x_nodes, val_nodes]:
        if is_continuous:
            score = to_rmse(x_hat_[nodes.type(torch.long)], x_all[nodes.type(torch.long)])
        else:
            score = to_recall(x_hat_[nodes.type(torch.long)], x_all[nodes.type(torch.long)], k)
        out_list.append(score)
    return out_list

def is_better(curr_acc_, best_acc_,is_continuous):
    if is_continuous:
        return curr_acc_ <= best_acc_
    else:
        return curr_acc_ >= best_acc_

@torch.no_grad()
def evaluate_last(dataset_name, model, edge_index, test_nodes, true_features,is_continuous, data, graphs):
    """
    Evaluate a prediction model after the training is done.
    """
    model.eval()
    x_hat, _ = model(edge_index, data, graphs)
    x_hat = x_hat[test_nodes]
    x_true = true_features[test_nodes]

    if is_continuous:
        return [to_r2(x_hat, x_true), to_rmse(x_hat, x_true), x_hat]
    else:
        k_list = [3, 5, 10] if dataset_name == 'steam' else [10, 20, 50]
        scores = []
        for k in k_list:
            scores.append(to_recall(x_hat, x_true, k))
        return scores












