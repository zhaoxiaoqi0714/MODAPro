import pandas as pd
import argparse
import torch
from torch import nn
from torch_geometric.utils import get_laplacian
from torch_sparse import SparseTensor

from sklearn.metrics import f1_score,mean_squared_error, mean_absolute_error, r2_score
import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import json

from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio

import warnings
warnings.filterwarnings("ignore")

class Args:
    pass
def Parse_args():
    with open('../MODAPro/Params.txt', 'r') as file:
        params = json.load(file)
    args = Args()
    for key, value in params.items():
        setattr(args, key, value)
    return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

def obtain_heterograph_struc(edge,node):
    node_type = list(set(node['type_index']))
    edge_type = list(set(edge['edge_type']))

    # reindex based on node type
    node['node_type_index'] = 0
    for ntype in node_type: node['node_type_index'][node['type_index'] == ntype] = range(node[node['type_index'] == ntype].shape[0])
    index_to_typeindex = dict(zip(node['node_index'],node['node_type_index']))

    edge_dict = {}
    for type in edge_type:
        tar_edge = edge[edge['edge_type'] == type]
        src_type, tar_type = type.split('_')
        src_index, tar_index = list(tar_edge['source'].map(index_to_typeindex)), list(tar_edge['target'].map(index_to_typeindex))
        edge_dict.update({
            (src_type,type,tar_type): (src_index, tar_index)
        })
    graph = dgl.heterograph(edge_dict)

    return graph, node


def get_graph(dataset_folder, is_weight=False):
    print('generating graph ...')
    edge_file = pd.read_table(dataset_folder + '/edges.txt', encoding='utf-8')
    node_feat = pd.read_csv(dataset_folder + '/mol_ml_feat.csv', encoding='utf-8', index_col=0)
    node_file = pd.read_table(dataset_folder + '/nodes.txt', encoding='utf-8')

    # condtructed graph
    graph, node_file = obtain_heterograph_struc(edge_file,node_file)
    print(graph)

    # edge weight [这里先没有设置weight，后面有需要的话填写在这]
    # if is_weight:
    #     graph.edges['r-i'].data['weight'] = torch.FloatTensor(r_i_edge_weight)
    #     graph.edges['i-r'].data['weight'] = torch.FloatTensor(r_i_edge_weight)
    #     graph.edges['r-r'].data['weight'] = torch.FloatTensor(recipe_edge_weight)
    #     graph.edges['i-i'].data['weight'] = torch.FloatTensor(ingr
    #     e_edge_weight)
    #     graph.edges['u-r'].data['weight'] = torch.FloatTensor(u_rate_r_edge_weight)
    #     graph.edges['r-u'].data['weight'] = torch.FloatTensor(u_rate_r_edge_weight)

    ## node features
    # tidying node feature file
    nodename_to_index= dict(zip(node_file['node'],node_file['node_index']))
    nodename_to_typeindex= dict(zip(node_file['node_index'],node_file['node_type_index']))
    node_feat.index = node_feat.index.map(nodename_to_index)
    node_feat = node_feat[~pd.isna(node_feat.index)]

    # add node feature into heterograph
    node_type_list = graph.ntypes
    for n_type in node_type_list:
        tar_node_index = node_file['node_index'][node_file['type'] == n_type]
        tar_node_feat = node_feat[node_feat.index.isin(tar_node_index)]
        tar_no_feat_node = tar_node_index[~tar_node_index.isin(tar_node_feat.index)]
        tar_nofeat_dframe = pd.DataFrame(np.zeros((len(tar_no_feat_node), tar_node_feat.shape[1])),index=tar_no_feat_node, columns=tar_node_feat.columns)
        tar_node_feat = pd.concat([tar_node_feat,tar_nofeat_dframe])
        tar_node_feat.index = tar_node_feat.index.map(nodename_to_typeindex)
        tar_node_feat = tar_node_feat.sort_index()

        # add info
        assert graph.num_nodes(n_type) == tar_node_feat.shape[0], 'The number of targeted type nodes do not match the features of targeted type nodes!'
        graph.nodes[n_type].data['feat'] = torch.tensor(np.array(tar_node_feat))
        graph.nodes[n_type].data['score'] = torch.tensor(tar_node_feat.sum(1))

    # 先不切分训练/验证/测试集，后面再划分

    return graph


def score(logits, labels, is_continuous):
    if not is_continuous:
        _, indices = torch.max(logits, dim=1)
        prediction = indices.long().cpu().numpy()
        labels = labels.cpu().numpy()
        score1 = (prediction == labels).sum() / len(prediction)  # accuracy
        score2 = f1_score(labels, prediction, average='micro')  # micro_f1
        score3 = f1_score(labels, prediction, average='macro')  # macro_f1
    else:
        prediction = logits.detach().numpy()
        score1 = mean_squared_error(labels, prediction)  # mse
        score2 = mean_absolute_error(labels, prediction)  # mae
        score3 = r2_score(labels, prediction)  # r2

    return score1, score2, score3


def evaluate(model, g, features, labels, mask, loss_func, is_continuous, detail=False):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    score1, score2, score3 = score(logits[mask], labels[mask], is_continuous)
    return loss, score1, score2, score3

def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(model)
        elif loss > self.best_loss:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if loss <= self.best_loss:
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))

def str2bool(v):
    """
    Convert a string variable into a bool.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ['true']:
        return True
    elif v.lower() in ['false']:
        return False
    else:
        raise argparse.ArgumentTypeError()

def to_laplacian(edge_index, num_nodes):
    """
    Make a graph Laplacian term for the GMRF loss.
    """
    if isinstance(edge_index, SparseTensor):
        row = edge_index.storage.row()
        col = edge_index.storage.col()
        edge_index = torch.stack([row, col])
    edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)
    size = num_nodes, num_nodes
    return torch.sparse_coo_tensor(edge_index, edge_weight, size=size, device=edge_index.device)

def to_mean_loss(features, laplacian):
    """
    Compute the loss term that compares features of adjacent nodes.
    """
    return torch.bmm(features.t().unsqueeze(1), laplacian.matmul(features).t().unsqueeze(2)).view(-1)


class BernoulliLoss(nn.Module):
    """
    Loss term for the binary features.
    """

    def __init__(self, version='base'):
        """
        Class initializer.
        """
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.version = version

    def forward(self, input, target):
        """
        Run forward propagation.
        """
        assert (target < 0).sum() == 0
        if self.version == 'base':
            loss = self.loss(input, target)
        elif self.version == 'balanced':
            pos_ratio = (target > 0).float().mean()
            weight = torch.ones_like(target)
            weight[target > 0] = 1 / (2 * pos_ratio)
            weight[target == 0] = 1 / (2 * (1 - pos_ratio))
            loss = self.loss(input, target) * weight
        else:
            raise ValueError(self.version)
        return loss.mean()


class GMRFLoss(nn.Module):
    """
    Implementation of the GMRF loss.
    """

    def __init__(self, beta=1):
        """
        Class initializer.
        """
        super().__init__()
        self.cached_adj = None
        self.beta = beta

    def forward(self, features, edge_index):
        """
        Run forward propagation.
        """
        if self.cached_adj is None:
            self.cached_adj = to_laplacian(edge_index, features.size(0))

        num_nodes = features.size(0)
        hidden_dim = features.size(1)
        eye = torch.eye(hidden_dim, device=features.device)
        l1 = (eye + features.t().matmul(features) / self.beta).logdet()
        l2 = to_mean_loss(features, self.cached_adj).sum()
        return (l2 - l1 / 2) / num_nodes


class GMRFSamplingLoss(nn.Module):
    """
    Implementation of the GMRF loss without deterministic modeling.
    """

    def __init__(self, beta=1):
        """
        Class initializer.
        """
        super().__init__()
        self.cached_adj = None
        self.beta = beta

    def forward(self, z_mean, z_std, edge_index):
        """
        Run forward propagation.
        """
        if self.cached_adj is None:
            self.cached_adj = to_laplacian(edge_index, z_mean.size(0))

        device = edge_index.device
        num_nodes = z_mean.size(0)
        rank = z_std.size(1)
        var_mat = self.beta * torch.eye(num_nodes, device=device) + z_std.matmul(z_std.t())

        eye = torch.eye(rank, device=device)
        l1 = (eye + z_std.t().matmul(z_std) / self.beta).logdet()
        l2 = self.cached_adj.matmul(var_mat).diagonal().sum()
        l3 = to_mean_loss(z_mean, self.cached_adj).sum()

        return (l3 / z_mean.size(1) + l2 - l1) / (2 * num_nodes)