"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from scipy.stats import rankdata
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


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


def fit_predict(train_smiles, train_y, test_smiles, seed):
    n = len(train_smiles)
    smi = list(train_smiles) + list(test_smiles)
    X = featurize(smi).astype(np.float64)
    D = descriptors(smi)
    keep = D[:n].std(axis=0) > 1e-9
    D = D[:, keep]
    dmu, dsd = D[:n].mean(axis=0), D[:n].std(axis=0) + 1e-9
    D = np.clip((D - dmu) / dsd, -5.0, 5.0)
    gamma = 1.0 / D.shape[1]
    K = 0.5 * (tanimoto(X, X) + rbf(D, D, gamma))
    F = np.hstack([X, D])
    y = np.asarray(train_y, dtype=np.float64)
    y = (rankdata(y) - 0.5) / len(y)
    tr = np.arange(n)
    te = np.arange(n, len(smi))

    def svr(a, b, yy):
        mu, sd = yy.mean(), yy.std() + 1e-9
        m = SVR(kernel="precomputed", C=5.0, epsilon=0.05)
        m.fit(K[np.ix_(a, a)], (yy - mu) / sd)
        return m.predict(K[np.ix_(b, a)])

    def lgbm(a, b, yy):
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15,
                              min_child_samples=5, colsample_bytree=0.3, subsample=0.8,
                              subsample_freq=1, random_state=seed, n_jobs=8, verbose=-1)
        m.fit(F[a], yy)
        return m.predict(F[b])

    def rf(a, b, yy):
        m = RandomForestRegressor(n_estimators=300, max_features=0.3, min_samples_leaf=2,
                                  n_jobs=8, random_state=seed)
        m.fit(F[a], yy)
        return m.predict(F[b])

    models = [svr, lgbm, rf]

    def base_all(a, b, yy):
        return np.column_stack([m(a, b, yy) for m in models])

    oof = np.zeros((n, len(models)))
    for a, b in KFold(n_splits=5, shuffle=True, random_state=seed).split(tr):
        oof[b] = base_all(tr[a], tr[b], y[a])
    Z = np.column_stack([rankdata(oof[:, j]) / n for j in range(len(models))])
    meta = Ridge(alpha=1.0, positive=True).fit(Z, y)
    P = base_all(tr, te, y)
    Pz = np.column_stack([rankdata(P[:, j]) / len(te) for j in range(len(models))])
    return meta.predict(Pz)
