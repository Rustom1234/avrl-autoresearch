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
    X = featurize(train_smiles).astype(np.float64)
    Xt = featurize(test_smiles).astype(np.float64)
    D = descriptors(train_smiles)
    Dt = descriptors(test_smiles)
    keep = D.std(axis=0) > 1e-9
    D, Dt = D[:, keep], Dt[:, keep]
    dmu, dsd = D.mean(axis=0), D.std(axis=0) + 1e-9
    D = np.clip((D - dmu) / dsd, -5.0, 5.0)
    Dt = np.clip((Dt - dmu) / dsd, -5.0, 5.0)
    gamma = 1.0 / D.shape[1]
    K_train = 0.5 * (tanimoto(X, X) + rbf(D, D, gamma))
    K_test = 0.5 * (tanimoto(Xt, X) + rbf(Dt, D, gamma))
    y = np.asarray(train_y, dtype=np.float64)
    y = (rankdata(y) - 0.5) / len(y)
    mu, sd = y.mean(), y.std() + 1e-9
    yn = (y - mu) / sd
    XT = np.hstack([X, D])
    XTt = np.hstack([Xt, Dt])

    def run(tr, Kte, Xte):
        svr = SVR(kernel="precomputed", C=5.0, epsilon=0.05).fit(K_train[np.ix_(tr, tr)], yn[tr])
        lg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, colsample_bytree=0.3, subsample=0.8,
                               subsample_freq=1, num_leaves=15, min_child_samples=10, n_jobs=8,
                               random_state=seed, verbose=-1).fit(XT[tr], yn[tr])
        rf = RandomForestRegressor(n_estimators=500, max_features=0.3, min_samples_leaf=2, n_jobs=8,
                                   random_state=seed).fit(XT[tr], yn[tr])
        P = np.column_stack([svr.predict(Kte), lg.predict(Xte), rf.predict(Xte)])
        return np.column_stack([rankdata(P[:, j]) / len(P) for j in range(P.shape[1])])

    n = len(y)
    oof = np.zeros((n, 3))
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(np.arange(n)):
        oof[va] = run(tr, K_train[np.ix_(va, tr)], XT[va])
    meta = Ridge(alpha=1.0, positive=True).fit(oof, y)
    return run(np.arange(n), K_test, XTt) @ meta.coef_
