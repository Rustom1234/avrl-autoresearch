"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import lightgbm as lgb
from rdkit.Chem import Descriptors
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr

# Settings
N_BITS = 2048
RADIUS = 2
W_DESC = 0.6  # weight of the descriptor-based LightGBM rank; fingerprint ExtraTrees gets 1 - W_DESC


_DESC_CACHE = {}


def desc_features(smiles):
    rows = []
    for s in smiles:
        if s not in _DESC_CACHE:
            m = Chem.MolFromSmiles(s)
            _DESC_CACHE[s] = [f(m) for _, f in Descriptors.descList]
        rows.append(_DESC_CACHE[s])
    X = np.array(rows, dtype=np.float64)
    X[~np.isfinite(X)] = 0.0
    return np.clip(X, -1e6, 1e6)


def count_features(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def bit_fingerprints(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float64)


def tanimoto(A, B):
    inter = A @ B.T
    return inter / np.maximum(A.sum(1)[:, None] + B.sum(1)[None, :] - inter, 1e-9)


def gp_mean(K_tr, y, K_te, sigma2):
    mu = y.mean()
    alpha = np.linalg.solve(K_tr + sigma2 * np.eye(len(y)), y - mu)
    return mu + K_te @ alpha


def choose_sigma2(K, y, smiles):
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in smiles]
    groups = np.unique(scaffolds, return_inverse=True)[1]
    best_score, best_s2 = -np.inf, 0.3
    for s2 in (0.1, 0.3, 1.0):
        pred = np.zeros(len(y))
        for tr, va in GroupKFold(n_splits=5).split(K, y, groups):
            pred[va] = gp_mean(K[np.ix_(tr, tr)], y[tr], K[np.ix_(va, tr)], s2)
        score = spearmanr(pred, y).correlation
        if score > best_score:
            best_score, best_s2 = score, s2
    return best_s2


def tune_lgbm(X, y, smiles, seed):
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in smiles]
    groups = np.unique(scaffolds, return_inverse=True)[1]
    grid = [dict(num_leaves=nl, learning_rate=lr, n_estimators=ne, min_child_samples=mc, colsample_bytree=cs)
            for nl in (7, 15, 31) for lr in (0.02, 0.05) for ne in (200, 500) for mc in (5, 10, 20) for cs in (0.3, 0.6)]
    rng = np.random.RandomState(seed)
    configs = [grid[i] for i in rng.choice(len(grid), 8, replace=False)]
    best_score, best_cfg = -np.inf, configs[0]
    for cfg in configs:
        pred = np.zeros(len(y))
        for tr, va in GroupKFold(n_splits=4).split(X, y, groups):
            m = lgb.LGBMRegressor(subsample=0.8, subsample_freq=1, random_state=seed, n_jobs=8, verbose=-1, **cfg)
            m.fit(X[tr], y[tr])
            pred[va] = m.predict(X[va])
        score = spearmanr(pred, y).correlation
        if score > best_score:
            best_score, best_cfg = score, cfg
    return best_cfg


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)

    X_lgb_tr = np.hstack([desc_features(train_smiles), bit_fingerprints(train_smiles)])
    X_lgb_te = np.hstack([desc_features(test_smiles), bit_fingerprints(test_smiles)])
    cfg = tune_lgbm(X_lgb_tr, y, train_smiles, seed)
    lgbm = lgb.LGBMRegressor(subsample=0.8, subsample_freq=1, random_state=seed, n_jobs=8, verbose=-1, **cfg)
    lgbm.fit(X_lgb_tr, y)
    p_desc = lgbm.predict(X_lgb_te)

    X_tr, X_te = bit_fingerprints(train_smiles), bit_fingerprints(test_smiles)
    K_tr = tanimoto(X_tr, X_tr)
    s2 = choose_sigma2(K_tr, y, train_smiles)
    p_fp = gp_mean(K_tr, y, tanimoto(X_te, X_tr), s2)

    n = len(p_desc)
    return W_DESC * rankdata(p_desc) / n + (1 - W_DESC) * rankdata(p_fp) / n
