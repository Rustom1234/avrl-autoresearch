"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from catboost import CatBoostRegressor
from scipy.stats import spearmanr

# Settings
N_BITS = 2048
RADIUS = 2
ITERS = 800
DEPTH = 6
LR = 0.05


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fp = np.zeros(N_BITS, dtype=np.float32)
        for k, v in AllChem.GetHashedMorganFingerprint(m, RADIUS, nBits=N_BITS).GetNonzeroElements().items():
            fp[k] = v
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float32)
        rows.append(np.concatenate([fp, desc]))
    return np.nan_to_num(np.array(rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def base_features(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([Descriptors.TPSA(m), Descriptors.MolWt(m), Descriptors.NumHDonors(m), Descriptors.MolLogP(m)])
    return np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def top_desc(D, r, k=6):
    rho = np.nan_to_num(np.array([spearmanr(D[:, j], r)[0] for j in range(D.shape[1])]))
    return np.argsort(-np.abs(rho))[:k]


def pair_features(D, idx):
    cols = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            a, b = D[:, idx[i]], D[:, idx[j]]
            cols.append(a * b)
            cols.append(a / (b + 1e-3))
    return np.column_stack(cols)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    A_tr = np.c_[np.ones(len(y)), base_features(train_smiles)]
    A_te = np.c_[np.ones(len(test_smiles)), base_features(test_smiles)]
    coef = np.linalg.lstsq(A_tr, y, rcond=None)[0]
    model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                              random_seed=seed, verbose=0, thread_count=8)
    resid = y - A_tr @ coef
    X_tr = featurize(train_smiles)
    X_te = featurize(test_smiles)
    top = top_desc(X_tr[:, N_BITS:], resid)
    X_tr = np.c_[X_tr, pair_features(X_tr[:, N_BITS:], top)]
    X_te = np.c_[X_te, pair_features(X_te[:, N_BITS:], top)]
    model.fit(X_tr, resid)
    return model.predict(X_te) + A_te @ coef
