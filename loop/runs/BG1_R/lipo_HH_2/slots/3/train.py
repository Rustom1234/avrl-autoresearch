"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# Settings
N_BITS = 2048
RADIUS = 2
N_ESTIMATORS = 800
LEARNING_RATE = 0.05


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def tanimoto(A, B):
    dot = A @ B.T
    na = (A * A).sum(1)[:, None]
    nb = (B * B).sum(1)[None, :]
    return dot / np.maximum(na + nb - dot, 1e-9)


def base_predict(Xtr, ytr, Xte, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(ytr))
    nv = len(ytr) // 10
    va_i, tr_i = idx[:nv], idx[nv:]
    params = dict(objective="l1", metric="l1", learning_rate=0.03, n_jobs=8, random_state=seed, verbose=-1)
    es = lgb.LGBMRegressor(n_estimators=3000, **params)
    es.fit(Xtr[tr_i], ytr[tr_i], eval_set=[(Xtr[va_i], ytr[va_i])], eval_metric="l1",
           callbacks=[lgb.early_stopping(100, verbose=False)])
    lg = lgb.LGBMRegressor(n_estimators=max(1, int(es.best_iteration_ * 1.1)), **params)
    lg.fit(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=150, max_features=0.3, n_jobs=8, random_state=seed)
    rf.fit(Xtr, ytr)
    svr = SVR(kernel="precomputed", C=5.0, epsilon=0.1)
    svr.fit(tanimoto(Xtr, Xtr), ytr)
    return np.column_stack([lg.predict(Xte), rf.predict(Xte), svr.predict(tanimoto(Xte, Xtr))])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    Xte = featurize(test_smiles)
    oof = np.zeros((len(y), 3))
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
        oof[va] = base_predict(X[tr], y[tr], X[va], seed)
    meta = LinearRegression(positive=True).fit(oof, y)
    return meta.predict(base_predict(X, y, Xte, seed))
