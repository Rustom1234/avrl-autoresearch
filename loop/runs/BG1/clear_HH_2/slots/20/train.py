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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer

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


def mlp_predict(D_tr, C_tr, y, D_te, C_te, seed):
    qt = QuantileTransformer(output_distribution='normal')
    X_tr = np.hstack([qt.fit_transform(D_tr), np.log1p(C_tr)])
    X_te = np.hstack([qt.transform(D_te), np.log1p(C_te)])
    preds = []
    for k in range(5):
        net = MLPRegressor(hidden_layer_sizes=(256, 64), activation='relu', alpha=1e-2, early_stopping=True,
                           validation_fraction=0.1, max_iter=500, random_state=seed + k)
        net.fit(X_tr, y)
        preds.append(net.predict(X_te))
    return np.mean(preds, axis=0)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)

    lgbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, min_child_samples=10, random_state=seed, n_jobs=8, verbose=-1)
    D_tr, D_te = desc_features(train_smiles), desc_features(test_smiles)
    lgbm.fit(D_tr, y)
    p_desc = lgbm.predict(D_te)

    X_tr, X_te = bit_fingerprints(train_smiles), bit_fingerprints(test_smiles)
    K_tr = tanimoto(X_tr, X_tr)
    s2 = choose_sigma2(K_tr, y, train_smiles)
    p_fp = gp_mean(K_tr, y, tanimoto(X_te, X_tr), s2)

    C_tr, C_te = count_features(train_smiles), count_features(test_smiles)
    p_mlp = mlp_predict(D_tr, C_tr, y, D_te, C_te, seed)

    n = len(p_desc)
    return (rankdata(p_desc) + rankdata(p_fp) + rankdata(p_mlp)) / (3 * n)
