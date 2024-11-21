import io
import json
import time
import pickle

import numpy as np
import torch
import torch_geometric
from torch import optim
from torch_geometric.data import Data, HeteroData

from heterograph.geometric_dataset import *
from utils.utils import *
from utils.tools import *
from models.score_models import *
from models.embed_models import *
from models.model_step import *

start_time = time.time()
# Params
args = Parse_args()
split = 'random'
emb_norm = 'unit'
conv = 'gcn'
x_loss = 'gaussian'
x_type = 'diag'
FileDir = os.path.join('../MODAPro/data')
molList = args.molList
TypeIndex_to_Type = dict(zip(range(len(molList)),molList))
Type_to_TypeIndex = dict(zip(molList,range(len(molList))))
metapaths = args.metapaths
out = os.path.join('../MODAPro/result')
from datetime import datetime
date = datetime.now().strftime('%Y-%m-%d-%H')
if not os.path.exists(out):
    os.mkdir(out)
    print('Created dir for saving results!')

# load data
data = build_dataset(args.dataset, split=split, dir=FileDir, Type_to_TypeIndex=Type_to_TypeIndex,molList=molList, split_ratio=[args.split_ratio,1-args.split_ratio])
# prepared graph
metapaths_dict, graphs = prepared_graph(data, molList,metapaths)
if not args.Heterogeneous:
    graphs = graphs.to_homogeneous()


# trained model
assert args.embedding_step+args.extracted_step > 0, 'Please select at least one step!'
if args.embedding_step:
    out_path,formatted_date = graph_embedding(args, data, graphs, conv, x_type, x_loss, emb_norm, out)
else:
    out_path = out + '/' + args.dataset
    formatted_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        print(f"Save folder already exists.")
    save_ebedding_res(model=None, saved_model_path=None, args=args,
                      data=data, graphs=graphs, out_path=out_path)

## Extracted key nodes
# Params
root = out_path
num_nodes = data.x.shape[0]
num_ntypes = {}
for mol in molList:
    num_ntypes[mol] = torch.sum(data.mask[mol]).item()
meta_paths = obtained_etypes_dict(metapaths)
with open(os.path.join(root, 'graphs.pkl'), 'rb') as f:
    graphs = pickle.load(f)
with open(os.path.join(root, 'data.pkl'), 'rb') as f:
    data = pickle.load(f)

if args.Heterogeneous:
    g, trans_adj_list = constructed_dgl_graphs(num_ntypes, metapaths, graphs, meta_paths, args, out_path)
else:
    g, trans_adj_list = constructed_homo_graphs(data,num_nodes,out_path)

# Train model
if args.extracted_step:
    scores = extracted_kn_step(g,out_path, data, molList, trans_adj_list,meta_paths,
                               num_ntypes, args, formatted_date)
else:
    feats = torch.load(os.path.join(out_path, 'features.pth')).to(args.device)
    scores = feats.sum(dim=1).to(args.device)

torch.save(scores, os.path.join(out_path, 'scores.pth'))

print('Traning Finished!')

import torch
import os
import pandas as pd

from utils.utils import Parse_args
from utils.DC import get_last_subfolder, load_data, DC, DC_res, save_global_res


# Params
args = Parse_args()
root = os.path.join("../MODAPro/result", args.dataset)
ntypes = args.molList
ntypes_dict = dict(zip(range(len(ntypes)), ntypes))

# load data
scores, G, data = load_data(root, args)
# Detecting communities
coms = DC(G)
# Tidying results
nodes_community, community_scores = DC_res(coms,scores)
# extracted top_k community
topk_community = sorted(community_scores, key=community_scores.get, reverse=True)[:args.Top_com]

# save_res
save_global_res(nodes_community, ntypes_dict, data, scores, community_scores, topk_community, root)

end_time = time.time()
print('Finished Community detection! Timing: {}'.format(end_time - start_time))
