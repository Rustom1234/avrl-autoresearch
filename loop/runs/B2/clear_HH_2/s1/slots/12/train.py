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


DESC_SUBSET = ["MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors", "NumRotatableBonds", "FractionCSP3",
               "RingCount"]


def small_desc(smiles):
    names = [n for n, _ in Descriptors.descList]
    idx = [names.index(n) for n in DESC_SUBSET]
    return desc_features(smiles)[:, idx]


def sq_dist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0)


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
    return best_s2, best_score


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)

    lgbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, min_child_samples=10, random_state=seed, n_jobs=8, verbose=-1)
    lgbm.fit(desc_features(train_smiles), y)
    p_desc = lgbm.predict(desc_features(test_smiles))

    X_tr, X_te = bit_fingerprints(train_smiles), bit_fingerprints(test_smiles)
    K_tan_tr, K_tan_te = tanimoto(X_tr, X_tr), tanimoto(X_te, X_tr)
    Z_tr, Z_te = small_desc(train_smiles), small_desc(test_smiles)
    mu, sd = Z_tr.mean(0), Z_tr.std(0) + 1e-9
    Z_tr = np.clip((Z_tr - mu) / sd, -5, 5)
    Z_te = np.clip((Z_te - mu) / sd, -5, 5)
    D_tr, D_te = sq_dist(Z_tr, Z_tr), sq_dist(Z_te, Z_tr)
    best = None
    for gamma in (0.25, 1.0):
        K_tr = K_tan_tr * (0.5 + 0.5 * np.exp(-gamma * D_tr / len(DESC_SUBSET)))
        s2, score = choose_sigma2(K_tr, y, train_smiles)
        if best is None or score > best[0]:
            best = (score, gamma, s2)
    _, gamma, s2 = best
    K_tr = K_tan_tr * (0.5 + 0.5 * np.exp(-gamma * D_tr / len(DESC_SUBSET)))
    K_te = K_tan_te * (0.5 + 0.5 * np.exp(-gamma * D_te / len(DESC_SUBSET)))
    p_fp = gp_mean(K_tr, y, K_te, s2)

    n = len(p_desc)
    return W_DESC * rankdata(p_desc) / n + (1 - W_DESC) * rankdata(p_fp) / n
