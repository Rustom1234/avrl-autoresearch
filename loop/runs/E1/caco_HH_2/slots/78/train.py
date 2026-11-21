"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestRegressor

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
    X_tr = featurize(train_smiles)
    X_te = featurize(test_smiles)
    r = y - A_tr @ coef
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.3,
                               random_state=seed, n_jobs=8)
    rf.fit(X_tr, r)
    leaves_tr = rf.apply(X_tr)
    leaves_te = rf.apply(X_te)
    # weight[i, j] = sum over trees of 1/(leaf size) when test i and train j share a leaf
    W = np.zeros((len(X_te), len(X_tr)))
    for t in range(leaves_tr.shape[1]):
        cnt = np.bincount(leaves_tr[:, t])
        same = leaves_te[:, t][:, None] == leaves_tr[:, t][None, :]
        W += same / cnt[leaves_tr[:, t]][None, :]
    # weighted median of training residuals for each test molecule
    order = np.argsort(r)
    cw = np.cumsum(W[:, order], axis=1)
    idx = np.minimum((cw < 0.5 * cw[:, -1:]).sum(axis=1), len(r) - 1)
    med = r[order][idx]
    return med + A_te @ coef
