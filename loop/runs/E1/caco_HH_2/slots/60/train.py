"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from catboost import CatBoostRegressor
from sklearn.linear_model import HuberRegressor

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


def prior_predict(X_tr, y, X_te):
    # Average of three linear priors: OLS, OLS with pairwise products, Huber fit.
    n = len(y)
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-9
    Z_tr, Z_te = (X_tr - mu) / sd, (X_te - mu) / sd
    # 1. OLS on the four terms
    A_tr = np.c_[np.ones(n), Z_tr]
    A_te = np.c_[np.ones(len(Z_te)), Z_te]
    c1 = np.linalg.lstsq(A_tr, y, rcond=None)[0]
    # 2. OLS on the four terms plus pairwise products
    i, j = np.triu_indices(Z_tr.shape[1])
    B_tr = np.c_[A_tr, Z_tr[:, i] * Z_tr[:, j]]
    B_te = np.c_[A_te, Z_te[:, i] * Z_te[:, j]]
    c2 = np.linalg.lstsq(B_tr, y, rcond=None)[0]
    # 3. Huber linear fit on the four terms
    h = HuberRegressor(epsilon=1.35, max_iter=500).fit(Z_tr, y)
    p_tr = (A_tr @ c1 + B_tr @ c2 + h.predict(Z_tr)) / 3.0
    p_te = (A_te @ c1 + B_te @ c2 + h.predict(Z_te)) / 3.0
    return p_tr, p_te


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    p_tr, p_te = prior_predict(base_features(train_smiles), y, base_features(test_smiles))
    model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                              random_seed=seed, verbose=0, thread_count=8)
    model.fit(featurize(train_smiles), y - p_tr)
    return model.predict(featurize(test_smiles)) + p_te
