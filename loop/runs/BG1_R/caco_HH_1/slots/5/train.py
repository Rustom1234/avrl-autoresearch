"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  # noqa: F401 (kept for reference; not used below)
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


_CACHE = {}


def featurize_one(smi):
    if smi not in _CACHE:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
        m = Chem.MolFromSmiles(smi)
        fp = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6).astype(np.float32)
        _CACHE[smi] = np.concatenate([fp, desc])
    return _CACHE[smi]


def featurize(smiles):
    return np.array([featurize_one(s) for s in smiles], dtype=np.float32)


def tanimoto(A, B):
    inter = A @ B.T
    na = A.sum(axis=1)[:, None]
    nb = B.sum(axis=1)[None, :]
    return inter / np.maximum(na + nb - inter, 1e-9)


def base_predict(Xtr, ytr, Xte, seed):
    # Returns an (n_test, 3) matrix of predictions from three diverse base models.
    rf = RandomForestRegressor(n_estimators=300, max_features=0.3, min_samples_leaf=1,
                               random_state=seed, n_jobs=8)
    rf.fit(Xtr[:, :N_BITS], ytr)
    p_rf = rf.predict(Xte[:, :N_BITS])

    Btr = (Xtr[:, :N_BITS] > 0).astype(np.float64)
    Bte = (Xte[:, :N_BITS] > 0).astype(np.float64)
    svr = SVR(kernel='precomputed', C=5.0, epsilon=0.1)
    svr.fit(tanimoto(Btr, Btr), ytr)
    p_svr = svr.predict(tanimoto(Bte, Btr))

    lgb = LGBMRegressor(objective='l1', n_estimators=600, learning_rate=0.03, num_leaves=31,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
                        random_state=seed, n_jobs=8, verbose=-1)
    lgb.fit(Xtr[:, N_BITS:], ytr)
    p_lgb = lgb.predict(Xte[:, N_BITS:])

    return np.column_stack([p_rf, p_svr, p_lgb])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    Xte = featurize(test_smiles)

    # Out-of-fold base predictions on the training set for the meta-learner.
    oof = np.zeros((len(y), 3))
    for tr, va in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
        oof[va] = base_predict(X[tr], y[tr], X[va], seed)
    meta = LinearRegression(positive=True).fit(oof, y)

    # Refit base models on all training data and combine with learned weights.
    P = base_predict(X, y, Xte, seed)
    return meta.predict(P)
