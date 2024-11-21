import os
import dgl
import pickle
import torch
import pandas as pd
import networkx as nx
import demon as d
from sklearn.metrics.pairwise import cosine_similarity

def get_last_subfolder(path):
    subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    subfolders_index = [folder.split('_Repeat_')[-1] for folder in subfolders]
    if subfolders:
        max_index = max(range(len(subfolders_index)), key=lambda i: int(subfolders_index[i]))
        return subfolders[max_index]
    else:
        return None


def load_data(root, args):
    if args.embedding_step:
        feats = torch.load(os.path.join(root, 'Embedding', 'features.pth'))
        scores = torch.load(os.path.join(root, 'Embedding', 'scores.pth'))
        hg = dgl.load_graphs(os.path.join(root, 'Embedding', 'dgl_graphs.dgl'))[0]
        with open(os.path.join(root, 'Embedding', 'data.pkl'), 'rb') as f:
            data = pickle.load(f)
    else:
        with open(os.path.join(root, 'data.pkl'), 'rb') as f:
            data = pickle.load(f)
        scores = torch.load(os.path.join(root, 'scores.pth'))
        feats = data.x
        hg = dgl.load_graphs(os.path.join(root, 'dgl_graphs.dgl'))[0]

    g = dgl.to_homogeneous(hg[0])
    nx_g = dgl.to_networkx(g)
    edges = list(nx_g.edges())
    cosine_sim_matrix = cosine_similarity(feats.numpy(), feats.numpy())

    G = nx.Graph()
    for edge in edges:
        src = edge[0]
        dis = edge[1]
        if src not in G.nodes: G.add_node(src)
        if dis not in G.nodes: G.add_node(dis)
        G.add_edge(src, dis, weight=cosine_sim_matrix[src, dis])
        G.add_edge(dis, src, weight=cosine_sim_matrix[dis, src])

    print('The final graph covers {} nodes, and {} edges.'.format(G.number_of_nodes(), G.number_of_edges()))

    return scores, G, data

def DC(G):
    dm = d.Demon(graph=G, epsilon=0.25, min_community_size=3)
    coms = dm.execute()

    print('The numbers of community: {}'.format(len(coms)))
    for i in range(len(coms)):
        print('The community_{} covering {} nodes.'.format(i + 1, len(coms[i])))

    return coms

def DC_res(coms,scores):
    nodes_community = {}
    community = {}
    for i, com in enumerate(coms):
        nodeList = list(com)
        community['Community_' + str(i + 1)] = nodeList
        for node in nodeList:
            if node not in nodes_community.keys():
                nodes_community[node] = ['Community_' + str(i + 1)]
            else:
                nodes_community[node].append('Community_' + str(i + 1))
    community_scores = {}
    for k, v in community.items():
        community_scores[k] = torch.sum(scores[v]).item()

    return nodes_community, community_scores

def save_global_res(nodes_community, ntypes_dict, data, scores, community_scores, topk_community, root):
    # tidying nodes information
    nodeIndex = []
    nodeComList = []
    nodeIds = []
    nodeTypes = []
    nodeScores = []
    for i, k in enumerate(nodes_community):
        comList = ';'.join(nodes_community[k])
        nodeType = ntypes_dict[data.nodeType_mask[k].item()]
        nodeId = data.nodeName[k].item()
        score = scores[k].item()

        nodeIndex.append(k)
        nodeComList.append(comList)
        nodeIds.append(nodeId)
        nodeTypes.append(nodeType)
        nodeScores.append(score)

    nodeInfo = pd.DataFrame({
        'NodeIndex': nodeIndex,
        'NodeCom': nodeComList,
        'NodeID': nodeIds,
        'NodeTypes': nodeTypes,
        'NodeScores': nodeScores
    })

    # tidying community info
    comInfo = pd.DataFrame(list(community_scores.items()), columns=['Community', 'Score'])
    comInfo['key'] = comInfo['Community'].isin(topk_community)

    # save results
    nodeInfo.to_csv(os.path.join(root, 'NodeInfo.csv'), index=None)
    comInfo.to_csv(os.path.join(root, 'CommunityInfo.csv'), index=None)