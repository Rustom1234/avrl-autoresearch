"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from scipy.stats import rankdata
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import lightgbm as lgb
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestRegressor
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


KRR_LAMBDA = 0.1


def base_fit_predict(X, y, Xt, seed):
    # Tanimoto kernel ridge
    mu = y.mean()
    alpha = np.linalg.solve(tanimoto(X, X) + KRR_LAMBDA * np.eye(len(y)), y - mu)
    p_krr = tanimoto(Xt, X) @ alpha + mu
    # random forest
    rf = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=8, random_state=seed)
    p_rf = rf.fit(X, y).predict(Xt)
    # LightGBM
    gbm = lgb.LGBMRegressor(learning_rate=0.03, n_estimators=500, num_leaves=31,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
                            random_state=seed, n_jobs=8, verbose=-1)
    p_gbm = gbm.fit(X, y).predict(Xt)
    return np.column_stack([p_krr, p_rf, p_gbm])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles).astype(np.float64)
    Xt = featurize(test_smiles).astype(np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    y = (rankdata(y) - 0.5) / len(y)
    # out-of-fold base predictions on the training set
    oof = np.zeros((len(y), 3))
    for tr, va in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
        oof[va] = base_fit_predict(X[tr], y[tr], X[va], seed)
    w, _ = nnls(oof, y)
    # refit base models on all training data and combine with learned weights
    return base_fit_predict(X, y, Xt, seed) @ w
