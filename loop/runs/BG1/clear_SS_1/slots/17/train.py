"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from rdkit.Chem.Scaffolds import MurckoScaffold
import lightgbm as lgb

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 500
MAX_FEATURES = "sqrt"
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def mixed_features(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        d = [f(m) for _, f in Descriptors.descList]
        rows.append(np.concatenate([np.array(d, dtype=np.float64), gen.GetCountFingerprintAsNumPy(m)]))
    X = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def tanimoto(A, B):
    inter = A @ B.T
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-9)


def base_ranks(Xtr, Mtr, y, Xte, Mte, seed, n_trees):
    rf = RandomForestRegressor(n_estimators=n_trees, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                               n_jobs=-1, random_state=seed).fit(Mtr, y)
    gb = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.3, min_child_samples=5, random_state=seed, n_jobs=8,
                           verbose=-1).fit(Xtr, y)
    svr = SVR(kernel="precomputed", C=5.0).fit(tanimoto(Xtr, Xtr), y)
    preds = [rf.predict(Mte), gb.predict(Xte), svr.predict(tanimoto(Xte, Xtr))]
    return np.column_stack([rankdata(p) / len(p) for p in preds])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    train_smiles = list(train_smiles)
    X, M = featurize(train_smiles), mixed_features(train_smiles)
    Xte, Mte = featurize(test_smiles), mixed_features(test_smiles)
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    groups = [MurckoScaffold.MurckoScaffoldSmiles(s) for s in train_smiles]
    oof = np.zeros((len(y), 3))
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        oof[va] = base_ranks(X[tr], M[tr], y[tr], X[va], M[va], seed, 200)
    w = LinearRegression(positive=True).fit(oof, y).coef_
    if w.sum() <= 0:
        w = np.ones(3)
    w = w / w.sum()
    return base_ranks(X, M, y, Xte, Mte, seed, N_TREES) @ w
