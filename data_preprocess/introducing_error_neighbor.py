import networkx as nx
import random
import os
import pandas as pd

def introduce_error_neighbors(graph, error_ratio):
    # 创建图的副本以免修改原始图
    modified_graph = graph.copy()

    # 计算需要引入的错误边的数量
    num_errors = int(graph.number_of_edges() * error_ratio)

    # 获取图中所有可能的边
    possible_edges = list(graph.edges())

    # 引入错误边
    for _ in range(num_errors):
        # 从可能的边中随机选择一个边
        edge_to_error = random.choice(possible_edges)

        # 从错误边的一个节点开始选择一个不在边上的节点
        new_neighbor = random.choice(list(set(graph.nodes()) - set(edge_to_error)))

        # 引入错误边
        modified_graph.add_edge(edge_to_error[0], new_neighbor)

    return modified_graph


# load original_dat
FilePath = r'E:\Python_project\MODA\MODAPro\data\PRAD\data_process'
edges = pd.read_csv(os.path.join(FilePath,'Certained_edges.csv'), index_col=0)
nodes = set(edges['source'].unique()) | set(edges['target'].unique())

# generated G
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges[['source', 'target']].itertuples(index=False, name=None))

# introduced error neighbor
error_ratios = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for ratio in error_ratios:
    modified_graph = introduce_error_neighbors(G, ratio)
    modified_edges = modified_graph.edges()
    modified_edges = pd.DataFrame(modified_edges, columns=['source', 'target'])
    modified_edges['edge_type'] = modified_edges.apply(lambda row: f"{row['source'].split('_')[1]}_{row['target'].split('_')[1]}", axis=1)

    modified_edges.to_csv(os.path.join(FilePath,'Certained_edges_'+str(ratio)+'error_neighbor.csv'), index=None)
