import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from tools.utils import *

# Params
args = Parse_args()

# load edge information
edges = pd.read_csv('../database/biological_network/Raw_Edges.csv')
graph = nx.from_pandas_edgelist(edges, 'Source', 'Target')

### calculating index of network
if not os.path.exists('../data/'+args.dataset+'/Plotting/'):
    os.makedirs('../data/'+args.dataset+'/Plotting/')
## Clearing: removing high degrees node
## degree
degree = pd.DataFrame(graph.degree(), columns=['node','degree'])
plot_degree_distribution(degree, args.dataset, 'RawNetwork_Degree',bw=0.5)

## Constructed certained disease network
feat = pd.read_csv('../data/'+args.dataset+'/data_process/molecules_ML_feat.csv',index_col=0)
feat['Type'] = feat['Type'].replace(['Enzyme','Gene'], 'Protein')
feat.index = feat.index.astype(str) + '_' + feat['Type']
# extracted edges based on targeted molecules
ex_edges = edges[(edges['Source_Type'].isin(args.MolType)) & (edges['Target_Type'].isin(args.MolType))]
ex_edges = replace_mol(ex_edges)
ex_edges['edge_type'] = ex_edges['Source_Type'] + '_' + ex_edges['Target_Type']

# matched the same edge types
ex_edges = transform_edge_types(ex_edges)
# reindex edges and nodes
edges = ex_edges[['source','target','edge_type','Index']]
edges = reindexed_nodes(edges)

## extracted the max sub-graph
graph = nx.from_pandas_edgelist(edges, 'source_encoded', 'target_encoded')
is_connected = nx.is_connected(graph)
print('The connectivity of graph: {}'.format(is_connected))

if args.extracted_connected:
    if not is_connected:
        graph, edges = handling_unconnected_graph(graph,edges)

# extracted feat which nodes in ex_edges
edges = edges.applymap(lambda x: x.lstrip() if isinstance(x, str) else x)
all_node_list = list(set(pd.concat([edges['source'],edges['target']])))
ex_feat = feat[(feat.index).isin(all_node_list)]
# ex_feat.to_csv('../data/'+args.dataset+'/data_process/molecules_ML_feat.csv', encoding='utf-8')

# extracted sub-graph based on the 2-hop neighbor of input nodes
InputNodeList = obtained_node_Pairs(ex_feat,edges)

# Get the multihop subgraph
subgraph = get_multihop_subgraph(graph, InputNodeList, args.num_hops)
SubNodes = list(subgraph.nodes)
SubEdges = edges[(edges['source_encoded'].isin(SubNodes)) & (edges['target_encoded'].isin(SubNodes))]
print('After extracting sub-graph based on the {}-hop neighbors of input nodes, the sub-graph have {} nodes and {} edges'.format(
    args.num_hops, len(SubNodes), SubEdges.shape[0]
))

# general final files
SubEdges[['source_id','source_type']] = SubEdges['source'].str.split('_', 1, expand=True)
SubEdges[['target_id','target_type']] = SubEdges['target'].str.split('_', 1, expand=True)
SubEdges['source_id'] = SubEdges['source_id'].astype(int)
SubEdges['target_id'] = SubEdges['source_id'].astype(str).str.split('.').str[0].astype(int)
SubEdges['source'] = SubEdges['source_id'].astype(int).astype(str) + "_" + SubEdges['source_type']
SubEdges['target'] = SubEdges['target_id'].astype(int).astype(str) + "_" + SubEdges['target_type']

allnodelist = list(set(pd.concat([SubEdges['source'],SubEdges['target']])))
edges = ex_edges[ex_edges['source'].isin(allnodelist) & ex_edges['target'].isin(allnodelist)]
feat = ex_feat[ex_feat.index.isin(allnodelist)]
subgraph = nx.from_pandas_edgelist(edges, 'source','target')

# extracted sub-graph by removing nodes which have higher degrees
degree = pd.DataFrame(subgraph.degree(), columns=['node','degree'])
degree = degree[degree['degree'] < args.ex_degree]
plot_degree_distribution(degree, args.dataset,'Disease-certained Network_degree', bw=0.5)

## retained the node and edge which node's degree < 200
edges = edges[edges['source'].isin(degree['node']) & edges['target'].isin(degree['node'])]
final_feat = feat[feat.index.isin(list(set(pd.concat([edges['source'],edges['target']]))))]
print('This biological network coverred {} edges, and final feat_dataframe included {} molecules.'.format(edges.shape[0], final_feat.shape[0]))

final_feat.to_csv('../data/'+args.dataset+'/data_process/molecules_ML_feat.csv', encoding='utf-8')
edges.to_csv('../data/'+args.dataset+'/data_process/Certained_edges.csv', encoding='utf-8')

type_counts = final_feat['Type'].value_counts()
print('The final feat covering .... ')
print(type_counts)