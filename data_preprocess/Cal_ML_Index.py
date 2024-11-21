import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from tools.ML import Combined_ML_index
from tools.utils import Parse_args

# Params
args = Parse_args()

# load file
# load clinical file
clinical = pd.read_csv(os.path.join('../data',args.dataset,args.clinical_file))
clinical['Sample_id'] = clinical['Sample_id'].astype(str)

if args.Sampling:
    normal_samples = clinical[clinical['group'] == 'Normal']
    tumor_samples = clinical[clinical['group'] == 'Tumor']
    tumor_samples_sampled = tumor_samples.sample(frac=args.Sampling_ratio, random_state=42)
    result_df = pd.concat([normal_samples, tumor_samples_sampled])
    clinical = result_df.reset_index(drop=True)

# load omics file
omics_FileList = args.omics_file
ML_indexes = {}
ML_types = {}
ML_molList = {}
for omic_File in tqdm(omics_FileList):
    profile = pd.read_csv(os.path.join('../data',args.dataset,omic_File))
    ## Fill nan
    data_to_fill = profile.iloc[:, 2:]
    row_means = data_to_fill.mean(axis=1)
    profile = profile.fillna(row_means, axis=0)

    sample_inter = list(set(map(str, profile.columns.tolist())) & set(map(str, clinical['Sample_id'])))
    selected_columns = profile.columns[:2].tolist() + sample_inter
    profile = profile.loc[:,selected_columns]
    clinical_tar = clinical.loc[clinical['Sample_id'].isin(sample_inter)]

    FileName = omic_File.split('.csv')[0]
    mol = profile.iloc[:,0].tolist()
    molType = profile.iloc[0,1]
    profile = profile.iloc[:,2:]
    profile = profile[[col for col in list(clinical['Sample_id']) if col in profile.columns]]

    # ML analysis
    x = np.array(profile).transpose()
    y = clinical.loc[clinical['Sample_id'].isin(profile.columns), 'group'].astype('category').cat.codes.values
    if args.scale:
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
    if np.isnan(x).any() or np.isinf(x).any() or np.max(np.abs(x)) > np.finfo(np.float64).max:
        # 找到 NaN 或 infinity 的位置
        nan_indices = np.isnan(x)
        inf_indices = np.isinf(x)
        col_means = np.nanmean(x, axis=0)
        x[nan_indices] = np.take(col_means, np.where(nan_indices)[1])
        x[inf_indices] = np.take(col_means, np.where(inf_indices)[1])

    ML_index = Combined_ML_index(x, y, args)
    if args.MLscale:
        scaler = MinMaxScaler()
        ML_index = pd.DataFrame(scaler.fit_transform(ML_index), columns=ML_index.columns)
    ML_index = ML_index.loc[:,[args.ANOVA, args.LDA, args.RF, args.SVM, args.Logit, args.Tree, args.GMM, args.KNN, args.PLSDA, args.Xgboost, args.Boruta, args.RFE]]
    ML_index.index = mol
    ML_index['Type'] = molType
    ML_indexes[FileName] = ML_index

print('Finished ML analysis.')

### Combined data
first_file, ML_res = next(iter(ML_indexes.items()))
iterator = iter(ML_indexes.items())
next(iterator)  # 跳过第一个键值对
for key, value in iterator:
    print('Add omics profile from {}'.format(key))
    ML_res = pd.concat([ML_res, value])

ML_res = ML_res.fillna(0)
ML_res.to_csv(os.path.join('../data',args.dataset,'ML_Attribution.csv'))

print('Finished ML matrix.')