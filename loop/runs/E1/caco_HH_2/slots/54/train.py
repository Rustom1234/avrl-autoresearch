"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

# Settings
N_SUB = 200


def desc_matrix(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([f(m) for _, f in Descriptors.descList])
    return np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def base_features(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([Descriptors.TPSA(m), Descriptors.MolWt(m), Descriptors.NumHDonors(m), Descriptors.MolLogP(m)])
    return np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    A_tr = np.c_[np.ones(len(y)), base_features(train_smiles)]
    A_te = np.c_[np.ones(len(test_smiles)), base_features(test_smiles)]
    coef = np.linalg.lstsq(A_tr, y, rcond=None)[0]
    D_tr = desc_matrix(train_smiles)
    D_te = desc_matrix(test_smiles)
    mu, sd = D_tr.mean(axis=0), D_tr.std(axis=0)
    keep = sd > 1e-6
    Z_tr = np.clip((D_tr[:, keep] - mu[keep]) / sd[keep], -5, 5)
    Z_te = np.clip((D_te[:, keep] - mu[keep]) / sd[keep], -5, 5)
    resid = y - A_tr @ coef
    sub = np.random.RandomState(seed).choice(len(y), min(N_SUB, len(y)), replace=False)
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, normalize_y=True, random_state=seed)
    gp.fit(Z_tr[sub], resid[sub])
    return gp.predict(Z_te) + A_te @ coef
