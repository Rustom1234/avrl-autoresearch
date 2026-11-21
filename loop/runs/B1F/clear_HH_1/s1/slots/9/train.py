"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from scipy.stats import rankdata, spearmanr
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.model_selection import KFold

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1
K_CHOICES = (20, 50, 100, None)


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def tanimoto(A, B):
    inter = A @ B.T
    na = A.sum(axis=1)[:, None]
    nb = B.sum(axis=1)[None, :]
    return inter / np.maximum(na + nb - inter, 1e-9)


def descriptors(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([f(m) for _, f in Descriptors.descList])
    D = np.array(rows, dtype=np.float64)
    return np.clip(np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)


def rbf(A, B, gamma):
    sq = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def _top_cols(D, y, k):
    if k is None:
        return np.arange(D.shape[1])
    R = np.column_stack([rankdata(D[:, j]) for j in range(D.shape[1])])
    R = R - R.mean(axis=0)
    ry = rankdata(y)
    ry = ry - ry.mean()
    r = (R * ry[:, None]).sum(axis=0) / (np.sqrt((R ** 2).sum(axis=0) * (ry ** 2).sum()) + 1e-12)
    return np.argsort(-np.abs(r))[:k]


def _svr_predict(T_tr, D_tr, y_tr, T_te, D_te):
    gamma = 1.0 / D_tr.shape[1]
    K_tr = 0.5 * (T_tr + rbf(D_tr, D_tr, gamma))
    K_te = 0.5 * (T_te + rbf(D_te, D_tr, gamma))
    mu, sd = y_tr.mean(), y_tr.std() + 1e-9
    model = SVR(kernel="precomputed", C=5.0, epsilon=0.05)
    model.fit(K_tr, (y_tr - mu) / sd)
    return model.predict(K_te)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles).astype(np.float64)
    Xt = featurize(test_smiles).astype(np.float64)
    D = descriptors(train_smiles)
    Dt = descriptors(test_smiles)
    keep = D.std(axis=0) > 1e-9
    D, Dt = D[:, keep], Dt[:, keep]
    dmu, dsd = D.mean(axis=0), D.std(axis=0) + 1e-9
    D = np.clip((D - dmu) / dsd, -5.0, 5.0)
    Dt = np.clip((Dt - dmu) / dsd, -5.0, 5.0)
    y = np.asarray(train_y, dtype=np.float64)
    y = (rankdata(y) - 0.5) / len(y)
    T, Tt = tanimoto(X, X), tanimoto(Xt, X)
    best_k, best_s = None, -np.inf
    for k in K_CHOICES:
        oof = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(D):
            cols = _top_cols(D[tr], y[tr], k)
            oof[te] = _svr_predict(T[np.ix_(tr, tr)], D[np.ix_(tr, cols)], y[tr],
                                   T[np.ix_(te, tr)], D[np.ix_(te, cols)])
        s = spearmanr(oof, y).correlation
        if s > best_s:
            best_k, best_s = k, s
    cols = _top_cols(D, y, best_k)
    return _svr_predict(T, D[:, cols], y, Tt, Dt[:, cols])
