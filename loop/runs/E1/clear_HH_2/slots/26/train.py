"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import itertools
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import lightgbm as lgb
from rdkit.Chem import Descriptors, MACCSkeys
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


def fp_sets(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    gens = [rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS),
            rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS),
            rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=N_BITS)]
    out = [np.array([g.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float64) for g in gens]
    out.append(np.array([list(MACCSkeys.GenMACCSKeys(m)) for m in mols], dtype=np.float64))
    return out


def tanimoto(A, B):
    inter = A @ B.T
    return inter / np.maximum(A.sum(1)[:, None] + B.sum(1)[None, :] - inter, 1e-9)


def gp_mean(K_tr, y, K_te, sigma2):
    mu = y.mean()
    alpha = np.linalg.solve(K_tr + sigma2 * np.eye(len(y)), y - mu)
    return mu + K_te @ alpha


def choose_kernel(Ks, y, smiles):
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in smiles]
    groups = np.unique(scaffolds, return_inverse=True)[1]
    folds = list(GroupKFold(n_splits=5).split(Ks[0], y, groups))
    best_score, best_cfg = -np.inf, ((1.0, 0.0, 0.0, 0.0), 0.3)
    for w in itertools.product((0.0, 0.5, 1.0), repeat=len(Ks)):
        if not any(w):
            continue
        K = sum(wi * Ki for wi, Ki in zip(w, Ks))
        for s2 in (0.1, 0.3, 1.0):
            pred = np.zeros(len(y))
            for tr, va in folds:
                pred[va] = gp_mean(K[np.ix_(tr, tr)], y[tr], K[np.ix_(va, tr)], s2)
            score = spearmanr(pred, y).correlation
            if score > best_score:
                best_score, best_cfg = score, (w, s2)
    return best_cfg


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)

    lgbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, min_child_samples=10, random_state=seed, n_jobs=8, verbose=-1)
    lgbm.fit(desc_features(train_smiles), y)
    p_desc = lgbm.predict(desc_features(test_smiles))

    F_tr, F_te = fp_sets(train_smiles), fp_sets(test_smiles)
    Ks_tr = [tanimoto(A, A) for A in F_tr]
    Ks_te = [tanimoto(B, A) for A, B in zip(F_tr, F_te)]
    w, s2 = choose_kernel(Ks_tr, y, train_smiles)
    K_tr = sum(wi * Ki for wi, Ki in zip(w, Ks_tr))
    K_te = sum(wi * Ki for wi, Ki in zip(w, Ks_te))
    p_fp = gp_mean(K_tr, y, K_te, s2)

    n = len(p_desc)
    return W_DESC * rankdata(p_desc) / n + (1 - W_DESC) * rankdata(p_fp) / n
