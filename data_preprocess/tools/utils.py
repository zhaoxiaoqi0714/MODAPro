import json
import time
from Bio import Entrez
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class Args:
    pass
def Parse_args():
    with open('../data/Params.txt', 'r') as file:
        params = json.load(file)
    args = Args()
    for key, value in params.items():
        setattr(args, key, value)
    return args

def symbol_to_geneid(symbol, max_retries=3):
    for _ in range(max_retries):
        try:
            Entrez.email = "zhaoxiaoqi0714@163.com"
            handle = Entrez.esearch(db="gene", term=f"{symbol}[Symbol]")
            record = Entrez.read(handle)
            handle.close()

            if "IdList" in record and record["IdList"]:
                gene_id = record["IdList"][0]
                return gene_id
            else:
                return None
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            time.sleep(2)  # Add a delay before retrying
    return None

def get_multihop_subgraph(graph, nodes, num_hops=2):
    """
    Extracts a multihop subgraph from the input graph around the given nodes.

    Parameters:
    - graph (NetworkX graph): The input graph.
    - nodes (list): List of nodes for which the subgraph will be extracted.
    - num_hops (int): Number of hops for neighbors. Default is 2.

    Returns:
    - NetworkX subgraph: The extracted subgraph.
    """

    # Initialize the set of nodes with the input nodes
    subgraph_nodes = set(nodes)

    # Iteratively add neighbors up to num_hops
    for _ in range(num_hops):
        neighbors = set()
        for node in subgraph_nodes:
            neighbors.update(graph.neighbors(node))
        subgraph_nodes.update(neighbors)

    # Extract the subgraph
    subgraph = graph.subgraph(subgraph_nodes)

    return subgraph


def plot_degree_distribution(data, dataset_name, file_name,bw=0.5):
    plt.figure(figsize=(6, 6))
    sns.set_style("ticks")
    sns.set_palette("muted")
    sns.kdeplot(data['degree'], shade=True, bw=bw)
    plt.xlim(0, data['degree'].max())
    plt.savefig(f'../data/{dataset_name}/Plotting/{file_name}.pdf')
    plt.show()


def transform_edge_types(data):
    etypes_list = list(set(data['edge_type']))
    etype_dict = {}

    for etype in etypes_list:
        if etype in etype_dict.keys():
            continue
        else:
            mol1 = etype.split("_")[0]
            mol2 = etype.split("_")[1]
            trans_etype = mol2 + '_' + mol1
            if trans_etype in etype_dict.keys():
                etype_dict[etype] = trans_etype
            else:
                etype_dict[etype] = etype

    data['edge_type'] = data['edge_type'].map(etype_dict)
    data['source'] = data['Source'].astype(int).astype(str) + '_' + data['Source_Type']
    data['target'] = data['Target'].astype(int).astype(str) + '_' + data['Target_Type']

    return data

def reindexed_nodes(data):
    label_encoder = LabelEncoder()
    nodes = data['source'].append(data['target'])
    nodes_encoded = label_encoder.fit_transform(nodes)

    data['source_encoded'] = nodes_encoded[:len(data)]
    data['target_encoded'] = nodes_encoded[len(data):]
    edges = data.sort_values(by=['source_encoded', 'target_encoded'], ascending=True)

    return edges

def handling_unconnected_graph(graph,edges):
    print('Handling the graph which is not connected ……')
    num_nodes = graph.number_of_nodes()
    # extracted the max sub-sub-graph
    connected_components = list(nx.connected_components(graph))
    largest_connected_component = max(connected_components, key=len)
    largest_connected_component_nodes = set(largest_connected_component)
    edges = edges[
        (edges['source_encoded'].isin(largest_connected_component_nodes)) &
        (edges['target_encoded'].isin(largest_connected_component_nodes))
        ]

    # reindex edges file
    edges = reindexed_nodes(edges)
    graph = nx.from_pandas_edgelist(edges, 'source_encoded', 'target_encoded')
    is_connected = nx.is_connected(graph)
    print('The connectivity of graph after extracting the max sub-graph: {}'.format(is_connected))

    return graph, edges

def replace_mol(edges):
    edges['Source_Type'] = edges['Source_Type'].replace(['Enzyme', 'Gene'], 'Protein')
    edges['Target_Type'] = edges['Target_Type'].replace(['Enzyme', 'Gene'], 'Protein')

    return edges

def obtained_node_Pairs(feat, edges):
    InputNodeList = list(set(feat.index))
    seen_pairs = set()
    encodes_dict = {}
    for index, row in edges.iterrows():
        src_pair = (row['source'], row['source_encoded'])
        dis_pair = (row['target'], row['target_encoded'])

        if src_pair not in seen_pairs:
            encodes_dict[row['source']] = row['source_encoded']
            seen_pairs.add(src_pair)

        if dis_pair not in seen_pairs:
            encodes_dict[row['target']] = row['target_encoded']
            seen_pairs.add(dis_pair)
    InputNodeList = [encodes_dict[item] for item in InputNodeList]

    return InputNodeList
