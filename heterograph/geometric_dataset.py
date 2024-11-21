import torch
import torch_geometric
import os
import numpy as np
import pickle
import pandas as pd
import dgl
from scipy.sparse import csc_matrix,csr_matrix
from torch_geometric.data import HeteroData
from utils.tools import *
from utils.utils import *
from models.embed_models import *

def build_dataset(dataset_name, split, dir,Type_to_TypeIndex,molList, split_ratio=[0.8, 0.2]):
    root = dir+'/'+dataset_name+'/data_process'
    dataset = process(root=root,Type_to_TypeIndex=Type_to_TypeIndex)
    dataset_split = split_data(dataset=dataset, split_ratio=split_ratio)
    dataset_split.mask = {}
    dataset_split.mask.update({mol: torch.tensor(check(value=dataset_split.nodeType_mask,
                                                       standard=Type_to_TypeIndex[mol]),
                                                 dtype=torch.bool) for mol in molList})
    return dataset_split

def sort_by_indices(values, indices):
    num_dims = indices.dim()
    new_shape = tuple(indices.shape) + tuple(
        1
        for _ in range(values.dim() - num_dims)
    )
    repeats = tuple(
        1
        for _ in range(num_dims)
    ) + tuple(values.shape[num_dims:])
    repeated_indices = indices.reshape(*new_shape).repeat(*repeats)
    return torch.gather(values, num_dims - 1, repeated_indices)

# dataset
def process(root,Type_to_TypeIndex):
    ## load file
    edges = pd.read_csv(root+'/Certained_edges.csv', index_col=0)
    k_feat = pd.read_csv(root+'/molecules_ML_feat.csv', index_col=0)
    set(edges['edge_type'])

    # reindex edges and nodes
    nodes = pd.DataFrame(pd.concat([edges['source'], edges['target']]))
    nodes.columns = ['Name']
    nodes = nodes.drop_duplicates(subset='Name')
    nodes[['NodeID', 'Types']] = nodes['Name'].str.split('_', 1, expand=True)
    TypeIndexDict = Type_to_TypeIndex

    nodes['TypesIndex'] = nodes['Types'].map(TypeIndexDict)
    nodes = nodes.sort_values(by='TypesIndex')
    nodes = nodes.reset_index()
    NameIndexDict = dict(zip(nodes['Name'], nodes.index))
    nodes = nodes.drop(columns=['index'])

    NodesDict = {}
    for index, tokens in nodes.iterrows():
        NodeDict = {
            'Name': tokens[0],
            'ID': tokens[1],
            'Index': index,
            'Types': tokens[2],
            'TypeIndex': tokens[3]
        }
        NodesDict[tokens[0]] = NodeDict

    # generated feat
    k_feat = k_feat.drop(columns=['Type'])
    k_feat = k_feat[(k_feat.index).isin(nodes['Name'])]
    k_feat = k_feat[~k_feat.index.duplicated(keep='first')]
    k_NodeList = list(k_feat.index)
    u_NodeList = list(nodes['Name'][~nodes['Name'].isin(k_feat.index)])
    u_feat = pd.DataFrame(0, index=u_NodeList, columns=list(k_feat.columns))
    feat = pd.concat([k_feat, u_feat])

    feat.index = feat.index.map(NameIndexDict)
    feat = feat.sort_index()
    labels = np.array(feat.sum(axis=1))

    assert feat.shape[0] == nodes.shape[
        0], 'The number of input nodes is not equal to the number of nodes in disease-certained network!'

    ## tidying final data
    edges['sourceIndex'] = edges['source'].map(NameIndexDict)
    edges['targetIndex'] = edges['target'].map(NameIndexDict)
    edges = edges.sort_values(by=['sourceIndex','targetIndex'])
    edge_index = np.vstack((np.array(edges['sourceIndex']), np.array(edges['targetIndex'])))

    nodes['feat'] = 1
    nodes['feat'][nodes['Name'].isin(u_NodeList)] = 0

    # combined data
    data = torch_geometric.data.Data(x=torch.tensor(np.array(feat),dtype=torch.float),
                                     edge_index=torch.tensor(edge_index,dtype=torch.long),
                                     y=torch.tensor(np.array(labels),dtype=torch.float))


    data.feat_mask = torch.tensor(np.array(nodes['feat']==1), dtype=torch.bool)
    data.nodeType_mask = torch.tensor(np.array(nodes['TypesIndex']).astype(np.int32), dtype=torch.int32)
    data.nodeName = torch.tensor(np.array(nodes['NodeID']).astype(np.int32), dtype=torch.int32)

    return data

def prepared_graph(data, molList,metapaths):
    metapaths_dict = constructed_metapaths(metapaths, data)
    print('generating graph ...')
    graphs = HeteroData()
    # add nodes and meta paths
    for mol in molList:
        graphs[mol].x = data.x[data.mask[mol]]
    for etype in metapaths:
        mol1 = etype.split("_")[0]
        mol2 = etype.split("_")[1]
        edgeIndex = data.edge_index[:, metapaths_dict[etype]]

        graphs[eval("'" + mol1 + "','" + etype + "','" + mol2 + "'")].edge_index = edgeIndex

    print('graph: ', graphs)

    return metapaths_dict, graphs


def get_transition(given_hete_adjs, metapath_info):
    # make sure deg>0
    homo_adj_list = []
    for i in range(len(metapath_info)):
        adj = given_hete_adjs[metapath_info[i][0]]
        for etype in metapath_info[i][1:]:
            adj = adj.dot(given_hete_adjs[etype])
        homo_adj_list.append(csc_matrix(adj))
    return homo_adj_list

def split_data(dataset, split_ratio,split='random'):
    # 切片train, validation和test，并设置mask
    lbl_num = dataset.y.shape[0]
    dataset.train_mask = torch.BoolTensor([False] * lbl_num)
    dataset.val_mask = torch.BoolTensor([False] * lbl_num)
    dataset.test_mask = torch.BoolTensor([False] * lbl_num)
    if split == 'random':
        if split_ratio is None:
            print('split ratio is None')
            pass
        else:
            num_type = torch.max(dataset.nodeType_mask)
            idx_train_all = []
            idx_val_all = []
            idx_test_all = []
            for c in range(num_type+1):
                idx = ((dataset.nodeType_mask == c) * (dataset.feat_mask)).nonzero(as_tuple=False).view(-1)
                if len(idx) == 0:
                    num_train_per_class = 0
                    num_val_per_class = 0
                    print('The {} node all have no features'.format(c))
                else:
                    num_nodeType_feat = len(idx)
                    num_train_per_class = int(np.ceil(num_nodeType_feat * split_ratio[0]))
                    num_val_per_class = int(np.floor(num_nodeType_feat * split_ratio[1]))
                    idx_shuffle = sort_by_indices(idx, torch.randperm(len(idx)))
                    idx_train = idx_shuffle[:num_train_per_class]
                    idx_val = idx[~np.isin(idx, idx_train)]
                    assert len(idx_train)+len(idx_val) == len(idx)

                # testIdx --> all nodes without features
                idx_test = ((dataset.nodeType_mask == c) * (~dataset.feat_mask)).nonzero(as_tuple=False).view(-1)
                assert len(idx_test) >= 0
                print('[Class:{}] Train:{} | Val:{} | Test:{}'.format(c, num_train_per_class, num_val_per_class,
                                                                      len(idx_test)))
                if len(idx) != 0:
                    [idx_train_all.append(x) for x in idx_train.tolist()]
                    [idx_val_all.append(x) for x in idx_val.tolist()]
                    [idx_test_all.append(x) for x in idx_test.tolist()]
                else:
                    [idx_test_all.append(x) for x in idx_test.tolist()]

            assert len(idx_train_all)+len(idx_val_all)+len(idx_test_all) == lbl_num

            dataset.train_mask[idx_train_all] = True
            dataset.val_mask[idx_val_all] = True
            dataset.test_mask[idx_test_all] = True

        return dataset

def constructed_metapaths(metapaths,data):
    metapaths_dict = {}
    print("Constructed the information about meta pathways........")
    for metapath in metapaths:
        mol1 = metapath.split("_")[0]
        mol2 = metapath.split("_")[1]

        mol1_mask = data.mask[mol1]
        mol2_mask = data.mask[mol2]

        if mol1 == mol2:
            metapaths_dict[metapath] = mol1_mask[data.edge_index[0]] * mol1_mask[data.edge_index[1]]
        else:
            metapaths_dict[metapath] = mol1_mask[data.edge_index[0]] * mol2_mask[data.edge_index[1]] + mol2_mask[
                data.edge_index[0]] * mol1_mask[data.edge_index[1]]

        print(f"{metapath}: {metapaths_dict[metapath].sum()}")

    return metapaths_dict

def save_ebedding_res(model, saved_model_path, args, data, graphs,out_path):
    if args.embedding_step:
        model.load_state_dict(torch.load(saved_model_path))
        # save results
        with torch.no_grad():
            x_hat, _ = model(data.edge_index, data, graphs)

        torch.save(model.state_dict(), os.path.join(out_path, 'model.pth'))
        torch.save(x_hat, os.path.join(out_path, 'features.pth'))
        torch.save(torch.nonzero(data.train_mask).squeeze(), os.path.join(out_path, 'trn_nodes.pth'))
        torch.save(torch.nonzero(data.val_mask).squeeze(), os.path.join(out_path, 'val_node.pth'))
        torch.save(torch.nonzero(data.test_mask).squeeze(), os.path.join(out_path, 'test_nodes.pth'))

        # save graphs and data
        with open(os.path.join(out_path, 'graphs.pkl'), 'wb') as f:
            pickle.dump(graphs, f)
        with open(os.path.join(out_path, 'data.pkl'), 'wb') as f:
            pickle.dump(data, f)
    else:
        # save graphs and data
        with open(os.path.join(out_path, 'graphs.pkl'), 'wb') as f:
            pickle.dump(graphs, f)
        with open(os.path.join(out_path, 'data.pkl'), 'wb') as f:
            pickle.dump(data, f)

    print('Finished saving data.')

def obtained_etypes_dict(metapaths):
    meta_paths_dict = {}
    for etype in metapaths:
        mol1 = etype.split('_')[0]
        mol2 = etype.split('_')[1]

        if mol1 == mol2:
            if mol1 not in meta_paths_dict.keys():
                meta_paths_dict[mol1] = [etype]
            else:
                meta_paths_dict[mol1].append(etype)
        else:
            trans_etype = mol2 + '_' + mol1
            if mol1 not in meta_paths_dict.keys():
                meta_paths_dict[mol1] = [etype, trans_etype]
            else:
                [meta_paths_dict[mol1].append(m) for m in [etype, trans_etype]]

    meta_paths = []
    for k, v in meta_paths_dict.items():
        meta_paths.append(v)

    return meta_paths

def constructed_dgl_graphs(num_ntypes, metapaths,graphs, meta_paths, args, out_path):
    num_nodes_list = list(num_ntypes.values())
    adj_dicts = {}
    for etype in metapaths:
        mol1 = etype.split('_')[0]
        mol2 = etype.split('_')[1]

        if mol1 == mol2:
            mol_index = num_nodes_list.index(num_ntypes[mol1])
            if mol_index < len(num_nodes_list):
                tar_graphs = graphs[eval("'" + mol1 + "', '" + etype + "', '" + mol2 + "'")]['edge_index']
                tar_graphs_src = [item.item() for item in tar_graphs[0]]
                tar_graphs_dis = [item.item() for item in tar_graphs[1]]
                tar_adj = np.zeros((num_ntypes[mol1], num_ntypes[mol1]), dtype=int)

                for index in range(len(tar_graphs_src)):
                    src = tar_graphs_src[index] - sum(num_nodes_list[:mol_index])
                    dis = tar_graphs_dis[index] - sum(num_nodes_list[:mol_index])

                    if 0 <= src < num_ntypes[mol1] and 0 <= dis < num_ntypes[mol1]:
                        tar_adj[src, dis] = 1
                        tar_adj[dis, src] = 1

                tar_adj = csc_matrix(tar_adj)
                adj_dicts[etype] = tar_adj
        else:
            mol1_index = num_nodes_list.index(num_ntypes[mol1])
            mol2_index = num_nodes_list.index(num_ntypes[mol2])
            trans_etype = mol2 + "_" + mol1
            tar_graphs = graphs[eval("'" + mol1 + "', '" + etype + "', '" + mol2 + "'")]['edge_index']
            tar_graphs_src = [item.item() for item in tar_graphs[0]]
            tar_graphs_dis = [item.item() for item in tar_graphs[1]]
            tar_adj = np.zeros((num_ntypes[mol1], num_ntypes[mol2]), dtype=int)
            trans_adj = np.zeros((num_ntypes[mol2], num_ntypes[mol1]), dtype=int)

            for index in range(len(tar_graphs_src)):
                src = tar_graphs_src[index] - sum(num_nodes_list[:mol1_index])
                dis = tar_graphs_dis[index] - sum(num_nodes_list[:mol2_index])

                if 0 <= src < num_ntypes[mol1] and 0 <= dis < num_ntypes[mol2]:
                    tar_adj[src, dis] = 1
                    trans_adj[dis, src] = 1

            tar_adj = csc_matrix(tar_adj)
            trans_adj = csc_matrix(trans_adj)
            adj_dicts[etype] = tar_adj
            adj_dicts[trans_etype] = trans_adj

    trans_adj_list = get_transition(adj_dicts, meta_paths)

    # constructed heterograph
    hetero_dict = {}
    for k, v in adj_dicts.items():
        mol1 = k.split("_")[0]
        mol2 = k.split("_")[1]
        key = eval("'" + mol1 + "', '" + k + "', '" + mol2 + "'")
        hetero_dict[key] = adj_dicts[k].nonzero()

    g = dgl.heterograph(hetero_dict)
    # Set params for training
    g = g.to(args.device)
    dgl.save_graphs(os.path.join(out_path,'dgl_graphs.dgl'), g)

    return g,trans_adj_list

def constructed_homo_graphs(data,num_nodes,out_path):
    # generated dpl-graph
    edge_index_numpy = data.edge_index.numpy()
    # prepared adj
    adj = np.zeros((num_nodes, num_nodes), dtype=int)
    for index in range(edge_index_numpy.shape[1]):
        src = data.edge_index[0][index]
        dis = data.edge_index[1][index]
        adj[src, dis] = 1
        adj[dis, src] = 1

    sparse_adj = csr_matrix(adj)
    g = dgl.from_scipy(sparse_adj, eweight_name='weight')
    g.ndata['x'] = data.x

    tar_adj = csc_matrix(adj)
    trans_adj_list = [tar_adj]
    dgl.save_graphs(os.path.join(out_path, 'dgl_graphs.dgl'), g)

    return g, trans_adj_list



