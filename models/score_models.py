import torch
import argparse
import pickle as pkl
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from models.gat import GATConv
import random
import re
import copy


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128, is_continue=True):
        super(SemanticAttention, self).__init__()
        self.is_continue = is_continue

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.in_size = in_size

    def forward(self, z):
        w = self.project(z).mean(0)
        if not self.is_continue:  # (M, 1)
            beta = torch.softmax(w, dim=0)  # (M, 1)
            beta = beta.expand((z.shape[0],) + beta.shape)
        else:
            beta = w.expand((z.shape[0],) + torch.Size((self.in_size,)))

        return (beta * z)  # (N, D * K)


class HANLayer(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout, settings,num_nodes,
                 is_continue,args):
        super(HANLayer, self).__init__()
        self.args = args
        self.gat_layers = nn.ModuleList()
        self.num_nodes = num_nodes
        if args.Heterogeneous:
            for i in range(len(meta_paths)):
                self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                               dropout, dropout, activation=F.elu, settings=settings[i]))
        else:
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu, settings=settings[0]))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads, is_continue=is_continue)
        self.meta_paths = [tuple(meta_path)[0] if len(meta_path) == 1 else tuple(meta_path) for meta_path in meta_paths]

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        num_nodes_list = list(self.num_nodes.values())

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()

        if self.args.Heterogeneous:
            for i,k in enumerate(self.num_nodes):
                if isinstance(self.meta_paths[i], str):
                    new_g = dgl.metapath_reachable_graph(g, [self.meta_paths[i]])
                else:
                    new_g = dgl.metapath_reachable_graph(g, self.meta_paths[i])
                tar_h = h[sum(num_nodes_list[:i]):sum(num_nodes_list[:i+1])]
                semantic_embeddings.append((self.gat_layers[i](new_g, tar_h).flatten(1)))

                semantic_embeddings = torch.cat(semantic_embeddings, dim=0)  # (N, M, D * K)
        else:
            new_g = g
            tar_h = h
            semantic_embeddings.append((self.gat_layers[0](new_g, tar_h).flatten(1)))

            semantic_embeddings = torch.cat(semantic_embeddings, dim=0)  # (N, M, D * K)


        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout, settings, num_nodes, is_continue, args):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout, settings, num_nodes,
                     is_continue,args))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l - 1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
        return self.predict(h)
