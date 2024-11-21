import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from tools.utils import Parse_args, symbol_to_geneid

# Params
args = Parse_args()

# load biological database
edges = pd.read_csv('../database/biological_network/Raw_Edges.csv')
nodes = pd.read_csv('../database/molecular_information/Raw_Nodes.csv')
project_feat = pd.read_csv(os.path.join('../data', args.dataset, 'ML_Attribution.csv'), index_col=0)

# node id trans
NodeTypes = list(set(project_feat['Type']))
IdDict = {}

if not os.path.exists('../TemporaryStorage'):
    os.makedirs('../TemporaryStorage')

for NodeType in NodeTypes:
    if NodeType == 'Metabolite':
        if args.metabolite_inputID_type != 'Name' and args.metabolite_inputID_type != 'Pubchem_ID':
            nodes_filtered = nodes.dropna(subset=[args.metabolite_inputID_type])
            # constructed id map dict
            meta_id_map_dict = {}
            for index, row in nodes_filtered.iterrows():
                InputIDs = row[args.metabolite_inputID_type].split(';')
                OutputID = int(row['Edge_id'])
                for InputID in InputIDs:
                    meta_id_map_dict[InputID] = OutputID
            IdDict.update(meta_id_map_dict)
        elif args.metabolite_inputID_type == 'Pubchem_ID':
            meta_id_map_dict = dict(zip(project_feat.index.astype(int), project_feat.index.astype(int)))
            IdDict.update(meta_id_map_dict)
    elif NodeType == 'Exposme':
        if args.exposme_inputID_type != 'Name' and args.exposme_inputID_type != 'Pubchem_ID':
            nodes_filtered = nodes.dropna(subset=[args.exposme_inputID_type])
            # constructed id map dict
            exp_id_map_dict = {}
            for index, row in nodes_filtered.iterrows():
                InputIDs = row[args.exposme_inputID_type].split(';')
                OutputID = int(row['Edge_id'])
                for InputID in InputIDs:
                    exp_id_map_dict[InputID] = OutputID
            IdDict.update(exp_id_map_dict)
        elif args.exposme_inputID_type == 'Pubchem_ID':
            exp_id_map_dict = dict(zip(project_feat.index.astype(int), project_feat.index.astype(int)))
            IdDict.update(exp_id_map_dict)

    elif NodeType == 'Gene' or NodeType == 'Protein':
        if args.geneprotein_inputID_type != 'Name':
            if args.geneprotein_inputID_type == 'GeneID':
                gene_id_map_dict = dict(zip(project_feat.index, project_feat.index))
                IdDict.update(gene_id_map_dict)
            else:
                continue #? 后面加别的类型的id转换
        else:
            ## Using nodes information
            nodes_filtered = nodes.dropna(subset=['Gene_ID'])
            GeneProList = list(project_feat.index[project_feat['Type'].isin(['Gene','Protein'])])
            # constructed id map dict1
            gene_id_map_dict = {}
            for index, row in nodes_filtered.iterrows():
                InputIDs = row[args.geneprotein_inputID_type].split(';')
                OutputID = int(row['Edge_id'])
                for InputID in InputIDs:
                    if InputID in GeneProList:
                        gene_id_map_dict[InputID] = OutputID
            UnannoGeneProList = [item for item in GeneProList if item not in gene_id_map_dict]
            print('There are {} Gene/Protein would be annotated in BioPython.'.format(len(UnannoGeneProList)))

            ## Using BioPython
            for InputID in tqdm(UnannoGeneProList, desc="Processing IDs", unit="ID"):
                with open('../TemporaryStorage/id_trans.txt', 'a+') as file:
                    InputID = (InputID.split(',')[0]).split(';')[0]
                    gene_id = symbol_to_geneid(InputID)
                    if gene_id is not None:
                        file.write(f"{InputID}: {gene_id}\n")
                        # gene_id_map_dict.update({InputID: gene_id})
                file.close()

            ## read the id list and appended into all dict
            with open('../TemporaryStorage/id_trans.txt', 'r') as file:
                # rowiter
                for line in file:
                    InputID = line.split(':')[0]
                    OutputID = (line.split(': ')[1]).split('\n')[0]
                    gene_id_map_dict.update({InputID:OutputID})
                file.close()
            IdDict.update(gene_id_map_dict)

print('All obtained {} dict on the molecules.'.format(len(IdDict)))
# save id dict and delete temporary strorage
# save id dict
if not os.path.exists('../data/'+args.dataset+'/data_process/'):
    os.makedirs('../data/'+args.dataset+'/data_process/')
IdTransDict = pd.DataFrame(list(IdDict.items()), columns=['InputIDs', 'OutputIDs'])
IdTransDict.to_csv('../data/'+args.dataset+'/data_process/IdTransDict.csv', index=False, encoding='utf-8')

# delete temporary strorage
if os.path.exists("../TemporaryStorage/id_trans.txt"):
    os.remove("../TemporaryStorage/id_trans.txt")

# ID trans
updateIndex = project_feat.index.map(IdDict)
project_feat_update = project_feat[~updateIndex.isna()]
project_feat_update.index = project_feat_update.index.map(IdDict)
project_feat_update.index = project_feat_update.apply(lambda row: f"{row.name}_{row['Type']}", axis=1)
print('The number of molecules in original FeatData is {}, the updated is {} after id trans'.format(
    project_feat.shape[0], project_feat_update.shape[0]
))

### save
project_feat_update.to_csv('../data/'+args.dataset+'/data_process/molecules_ML_feat.csv', index=True, encoding='utf-8')