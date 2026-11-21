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


def base_preds(tr_smiles, tr_y, te_smiles, seed):
    lgbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, min_child_samples=10, random_state=seed, n_jobs=8, verbose=-1)
    lgbm.fit(desc_features(tr_smiles), tr_y)
    p_desc = lgbm.predict(desc_features(te_smiles))

    X_tr, X_te = bit_fingerprints(tr_smiles), bit_fingerprints(te_smiles)
    K_tr = tanimoto(X_tr, X_tr)
    s2 = choose_sigma2(K_tr, tr_y, tr_smiles)
    p_fp = gp_mean(K_tr, tr_y, tanimoto(X_te, X_tr), s2)
    return p_desc, p_fp


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    smi = np.asarray(train_smiles, dtype=object)

    # out-of-fold base predictions on the training data, grouped by Murcko scaffold
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in smi]
    groups = np.unique(scaffolds, return_inverse=True)[1]
    oof_d, oof_f = np.zeros(len(y)), np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=5).split(smi, y, groups):
        oof_d[va], oof_f[va] = base_preds(smi[tr], y[tr], smi[va], seed)
    A, B = rankdata(oof_d), rankdata(oof_f)
    C = (A + B) / 2
    comps = np.vstack([A, B, C])

    # grid search blend weights on the simplex (step 0.1) over (descriptor, GP, average)
    best_score, best_w = -np.inf, np.array([0.6, 0.4, 0.0])
    for i in range(11):
        for j in range(11 - i):
            w = np.array([i, j, 10 - i - j]) / 10.0
            score = spearmanr(w @ comps, y).correlation
            if score > best_score:
                best_score, best_w = score, w

    p_desc, p_fp = base_preds(smi, y, test_smiles, seed)
    n = len(p_desc)
    rd, rf = rankdata(p_desc) / n, rankdata(p_fp) / n
    comps_te = np.vstack([rd, rf, (rd + rf) / 2])
    return best_w @ comps_te
