import torch
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sample_y_nodes(num_nodes, y, y_ratio, seed):
    """
    Sample nodes with observed labels.
    """
    if y_ratio == 0:
        y_nodes = None
        y_labels = None
    elif y_ratio == 1:
        y_nodes = torch.arange(num_nodes)
        y_labels = y[y_nodes]
    else:
        y_nodes, _ = train_test_split(np.arange(num_nodes), train_size=y_ratio, random_state=seed,
                                      stratify=y.numpy())
        y_nodes = torch.from_numpy(y_nodes)
        y_labels = y[y_nodes]
    return y_nodes, y_labels


def split_sample(trn_size, node_fx, node_allx, seed):
    import random
    random.seed(seed)

    trn_nodes = random.sample(node_fx, k=round(trn_size * len(node_fx)))
    val_nodes = list(set(node_fx).difference(set(trn_nodes)))
    test_nodes = list(set(node_allx).difference(set(trn_nodes + val_nodes)))

    trn_nodes = torch.tensor(trn_nodes)
    val_nodes = torch.tensor(val_nodes)
    test_nodes = torch.tensor(test_nodes)
    assert len(trn_nodes) + len(val_nodes) + len(test_nodes) == len(
        node_allx), 'The error existed during spliting sample!'

    return trn_nodes, val_nodes, test_nodes

def print_log(epoch, loss_list, acc_list):
    """
    Print a log during the training.
    """
    print(f'{epoch:5d}', end=' ')
    print(' '.join(f'{e:.4f}' for e in loss_list), end=' ')
    print(' '.join(f'{e:.4f}' for e in acc_list))

def check(value, standard):
    res_list = []
    for i,n in enumerate(value):
        res_list.append(np.int(n) == np.int(standard))
    return res_list

def load_data(dataset, split, seed, normalize, verbose):
    if dataset == 'PRAD':
        # load file
        nodes = pd.read_table('data/' + dataset + '/nodes.txt', sep='\t', encoding='utf-8')
        edges = pd.read_table('data/' + dataset + '/edges.txt', sep='\t', encoding='utf-8')
        assert len(set(edges['source'].tolist() + edges['target'].tolist())) == nodes.shape[
            0], 'The number of nodes need match the number of nodes in edges file!'
        assert (pd.concat([edges['source'], edges['target']]).isin(
            nodes['node_index']) == 'False').sum() == 0, 'Some nodes do not match between nodes and edges file!'
        mol_feat = pd.read_csv('data/' + dataset + '/mol_ml_feat.csv', encoding='utf-8', index_col=0)
        mol_feat = mol_feat[mol_feat.index.isin(nodes['node'])]
        # mol_label = pd.read_csv('data/' + dataset + '/label.csv', encoding='utf-8', index_col=0)
        # mol_label = mol_label[mol_label.index.isin(nodes['node'])]
        mol_label = pd.DataFrame(mol_feat.apply(lambda x:x.sum(), axis = 1), columns=['label'])
        assert mol_feat.shape[0] == mol_label.shape[0], 'The label do not match the node feature!'

        # tidy data
        node_id_to_index = dict(zip(nodes['node'], nodes['node_index']))
        non_feat_node = nodes['node'][~nodes['node'].isin(mol_feat.index)].tolist()
        mol_nofeat = pd.DataFrame(np.zeros((len(non_feat_node), mol_feat.shape[1])), columns=mol_feat.columns,
                                  index=non_feat_node)
        mol_nolabel = pd.DataFrame(0, columns=['label'],
                                   index=non_feat_node)
        feat = pd.concat([mol_feat, mol_nofeat], axis=0)
        feat.index = feat.index.map(node_id_to_index).astype('int64')
        label = pd.concat([mol_label, mol_nolabel], axis=0)
        label.index = label.index.map(node_id_to_index).astype('int64')
        mol_label.index = mol_label.index.map(node_id_to_index).astype('int64')
        feat.sort_index(inplace=True)
        label.sort_index(inplace=True)
        mol_label.sort_index(inplace=True)

        # now do not consider heterograph
        edges = edges[['source', 'target']]

    node_fx = mol_label.index.tolist()
    node_allx = feat.index.tolist()
    node_y = torch.tensor(np.array(label['label'].tolist()))
    node_x = torch.tensor(np.array(feat))

    if torch.isnan(node_x).any():
        print('Feature tensor covered NaN, and replaced by zero')
        node_x = torch.where(torch.isnan(node_x), torch.full_like(node_x, 0), node_x)

    assert torch.isnan(node_x).any() == False, 'Feature tensor covered NaN!'

    edge_index = torch.tensor([edges['source'].tolist(),
                               edges['target'].tolist()])

    if normalize:
        assert (node_x < 0).sum() == 0  # all positive features
        norm_x = node_x.clone()
        norm_x[norm_x.sum(dim=1) == 0] = 1
        norm_x = norm_x / norm_x.sum(dim=1, keepdim=True)
        node_x = norm_x

    if split is not None and len(split) == 2 and sum(split) == 1:
        trn_nodes, val_nodes, test_nodes = split_sample(trn_size=split[0],node_fx=node_fx,node_allx=node_allx,seed=seed)
    else:
        raise ValueError(split)

    if verbose:
        print('Data:', dataset)
        print('Number of nodes:', node_x.size(0))
        print('Number of edges:', edges.shape[0])
        print('Number of features:', node_x.size(1))
        print('Ratio of nonzero features:', str(round((len(node_fx) / len(node_allx))*100,2)) + '%')
        print('Number of classes:', node_y.max().item() + 1 if node_y is not None else 0)

    return edge_index, node_x, node_y, trn_nodes, val_nodes, test_nodes

def to_device(gpu):
    """
    Return a PyTorch device from a GPU index.
    """
    if gpu is not None and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu}')
    else:
        return torch.device('cpu')

def plotting_train_res(out_path, filename, savename):
    plt_dat = pd.read_csv(os.path.join(out_path, filename), delim_whitespace=True)
    plt_dat = plt_dat[['epoch', 'trn', 'val']]
    plt_dat = pd.melt(plt_dat, id_vars=['epoch'], var_name='trn_val', value_name='value')

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plt_dat, x='epoch', y='value', hue='trn_val', style='trn_val',
                 markers={"trn": "o", "val": "s"})
    sns.scatterplot(data=plt_dat, x='epoch', y='value', hue='trn_val', style='trn_val',
                    markers={"trn": "o", "val": "s"})
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(os.path.join(out_path, f'{savename}.pdf'))






