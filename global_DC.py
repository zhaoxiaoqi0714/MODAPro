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

print('Finished Community detection!')