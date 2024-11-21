import os
import numpy as np
import pandas as pd
from scipy.stats import norm, nbinom
from scipy.linalg import toeplitz
from numpy.random import default_rng

def ZeroInflated(dist, pzero):
    return [0 if i == 0 else dist for i in range(2)]

def f(use_blocks, p, p_info):
    target_corrs = {}

    for corr in ["no_corr", "low_corr", "medium_corr", "high_corr"]:
        if corr == "no_corr":
            coef = 0
            cst = 0.0
        else:
            coef = 0.1
            if corr == "low_corr":
                cst = 0.2
            elif corr == "medium_corr":
                cst = 0.5
            elif corr == "high_corr":
                cst = 0.7

        base_corr = np.zeros((p, p))

        if use_blocks:
            i = 0
            while i < p - 4:
                base_corr[i:i+5, i:i+5] = coef * toeplitz(np.random.rand(5)) + cst
                i += 5
            target_corrs[corr] = base_corr
            np.fill_diagonal(target_corrs[corr], 1)
        else:
            base_corr[:p_info, :p_info] = coef * toeplitz(np.random.rand(p_info)) + cst
            target_corrs[corr] = base_corr
            np.fill_diagonal(target_corrs[corr], 1)

    margins = {
        "normal": [norm(0, 1) for _ in range(p)],
        "NB": [nbinom(2, 0.1) for _ in range(p)],
    }
    return target_corrs, margins

# p = int(input("Enter p: "))
# p_info = int(input("Enter p_info: "))
# marg = input("Enter marg (normal/NB): ")
# corr = input("Enter corr (no_corr/low_corr/medium_corr/high_corr): ")
# use_blocks = input("Enter use_blocks (yes/no): ").lower() == "yes"

p = 100
p_info = 0.05
marg = 'normal'
corr = 'medium_corr'
use_blocks = 'yes'

if marg == "normal" or marg == "ZI":
    marg = "normal"
elif marg == "NB" or marg == "ZINB":
    marg = "NB"

if marg == "normal":
    margin = [norm(0, 1) for _ in range(p)] # 表示均值（mean）为 0，标准差（standard deviation）为 1 的标准正态分布
else:
    margin = [nbinom(2, 0.1) for _ in range(p)]

if corr == "no_corr":
    coef = 0
    cst = 0.0
else:
    coef = 0.1
    if corr == "low_corr":
        cst = 0.2
    elif corr == "medium_corr":
        cst = 0.5
    elif corr == "high_corr":
        cst = 0.7

base_corr = np.zeros((p, p))
if use_blocks:
    i = 0
    while i < p - 4:
        base_corr[i:i+5, i:i+5] = coef * toeplitz(np.random.rand(5)) + cst
        i += 5
    np.fill_diagonal(base_corr, 1)
else:
    base_corr[:p_info, :p_info] = coef * toeplitz(np.random.rand(p_info)) + cst
    np.fill_diagonal(base_corr, 1)

target_corr = base_corr

dir_path = os.path.dirname(os.path.realpath(__file__))

if use_blocks:
    x = np.random.multivariate_normal(mean=np.zeros(p), cov=target_corr, size=50000)
    pd.DataFrame(x).to_csv(f"{dir_path}/Norta data {p_info} (block)/{p} feats {marg} {corr}.csv", index=False, header=True)
    rng = default_rng(42)
    rand_zero_one_mask = rng.integers(1, 11, size=x.shape) >= 10
    x[rand_zero_one_mask] = 0
    marg_zi = "ZI" if marg == "NB" else "ZINB"
    pd.DataFrame(x).to_csv(f"{dir_path}/Norta data {p_info} (block)/{p} feats {marg_zi} {corr}.csv", index=False, header=True)
else:
    x = np.random.multivariate_normal(mean=np.zeros(p), cov=target_corr, size=50000)
    pd.DataFrame(x).to_csv(f"{dir_path}/Norta data {p_info}/{p} feats {marg} {corr}.csv", index=False, header=True)
    rng = default_rng(42)
    rand_zero_one_mask = rng.integers(1, 11, size=x.shape) >= 10
    x[rand_zero_one_mask] = 0
    marg_zi = "ZI" if marg == "NB" else "ZINB"
    pd.DataFrame(x).to_csv(f"{dir_path}/Norta data {p_info}/{p} feats {marg_zi} {corr}.csv", index=False, header=True)
