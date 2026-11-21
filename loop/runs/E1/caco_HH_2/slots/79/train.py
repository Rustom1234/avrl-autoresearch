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
K = 10


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
    r = y - A_tr @ coef
    X_tr = featurize(train_smiles)
    X_te = featurize(test_smiles)
    # Tanimoto neighbours on binary Morgan bits (first N_BITS columns of featurize)
    B_tr = (X_tr[:, :N_BITS] > 0).astype(np.float32)
    B_te = (X_te[:, :N_BITS] > 0).astype(np.float32)
    n_tr = B_tr.sum(1)
    n_te = B_te.sum(1)
    I_tr = B_tr @ B_tr.T
    S_tr = I_tr / (n_tr[:, None] + n_tr[None, :] - I_tr + 1e-9)
    np.fill_diagonal(S_tr, -1.0)
    I_te = B_te @ B_tr.T
    S_te = I_te / (n_te[:, None] + n_tr[None, :] - I_te + 1e-9)
    nn_tr = np.argsort(-S_tr, axis=1)[:, :K]
    nn_te = np.argsort(-S_te, axis=1)[:, :K]
    # Training pairs (i, j): target is the difference of residuals
    i_idx = np.repeat(np.arange(len(y)), K)
    j_idx = nn_tr.ravel()
    Xi, Xj = X_tr[i_idx], X_tr[j_idx]
    P = np.hstack([Xi, Xj, Xi - Xj])
    d = r[i_idx] - r[j_idx]
    model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                              random_seed=seed, verbose=0, thread_count=8)
    model.fit(P, d)
    # Test: average over anchors of (anchor residual + predicted difference)
    t_idx = np.repeat(np.arange(len(test_smiles)), K)
    a_idx = nn_te.ravel()
    Xt, Xa = X_te[t_idx], X_tr[a_idx]
    Q = np.hstack([Xt, Xa, Xt - Xa])
    f = model.predict(Q).reshape(-1, K)
    return A_te @ coef + (r[nn_te] + f).mean(1)
