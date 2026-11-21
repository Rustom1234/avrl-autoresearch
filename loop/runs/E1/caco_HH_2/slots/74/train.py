"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from catboost import CatBoostRegressor

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


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    A_tr = np.c_[np.ones(len(y)), base_features(train_smiles)]
    A_te = np.c_[np.ones(len(test_smiles)), base_features(test_smiles)]
    coef = np.linalg.lstsq(A_tr, y, rcond=None)[0]
    model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                              random_seed=seed, verbose=0, thread_count=8)
    X_tr = featurize(train_smiles)
    X_te = featurize(test_smiles)
    resid = y - A_tr @ coef
    model.fit(X_tr, resid)
    cb = model.predict(X_te)
    # kNN median of OLS residuals in standardised, winsorised RDKit descriptor space
    D_tr = X_tr[:, N_BITS:].astype(np.float64)
    D_te = X_te[:, N_BITS:].astype(np.float64)
    lo, hi = np.percentile(D_tr, 1, axis=0), np.percentile(D_tr, 99, axis=0)
    D_tr = np.clip(D_tr, lo, hi)
    D_te = np.clip(D_te, lo, hi)
    mu, sd = D_tr.mean(0), D_tr.std(0)
    sd[sd < 1e-9] = 1.0
    Z_tr = (D_tr - mu) / sd
    Z_te = (D_te - mu) / sd
    d2 = (Z_te ** 2).sum(1)[:, None] + (Z_tr ** 2).sum(1)[None, :] - 2 * Z_te @ Z_tr.T
    idx = np.argsort(d2, axis=1)[:, :15]
    knn = np.median(resid[idx], axis=1)
    return 0.75 * cb + 0.25 * knn + A_te @ coef
