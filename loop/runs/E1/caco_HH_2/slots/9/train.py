"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVR
import lightgbm as lgb

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


def tanimoto(A, B):
    dot = A @ B.T
    na = (A * A).sum(1)[:, None]
    nb = (B * B).sum(1)[None, :]
    return dot / np.maximum(na + nb - dot, 1e-9)


def base_predict(X, y, Xt, seed):
    F, Ft = X[:, :N_BITS], Xt[:, :N_BITS]
    rf = RandomForestRegressor(n_estimators=300, n_jobs=8, random_state=seed).fit(F, y)
    lg = lgb.LGBMRegressor(objective='l1', num_leaves=15, learning_rate=0.05, n_estimators=400,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                           n_jobs=8, random_state=seed, verbose=-1).fit(X, y)
    B, Bt = (F > 0).astype(np.float64), (Ft > 0).astype(np.float64)
    mu = y.mean()
    svr = SVR(kernel='precomputed', C=5.0, epsilon=0.1).fit(tanimoto(B, B), y - mu)
    return np.column_stack([rf.predict(Ft), lg.predict(Xt), svr.predict(tanimoto(Bt, B)) + mu])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    Xt = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) for s in train_smiles]
    codes = {}
    groups = np.array([codes.setdefault(s, len(codes)) for s in scaf])
    oof = np.zeros((len(y), 3))
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        oof[va] = base_predict(X[tr], y[tr], X[va], seed)
    res = minimize(lambda w: np.mean(np.abs(oof @ w - y)), np.ones(3) / 3, method='SLSQP',
                   bounds=[(0, 1)] * 3, constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}])
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    return base_predict(X, y, Xt, seed) @ w
