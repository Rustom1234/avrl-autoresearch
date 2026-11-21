"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from scipy.stats import rankdata
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold
from rdkit.Chem.Scaffolds import MurckoScaffold

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


def descriptors(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([f(m) for _, f in Descriptors.descList])
    D = np.array(rows, dtype=np.float64)
    return np.clip(np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)


def rbf(A, B, gamma):
    sq = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def scaffold_groups(smiles):
    ids = {}
    out = []
    for s in smiles:
        try:
            key = MurckoScaffold.MurckoScaffoldSmiles(smiles=s)
        except Exception:
            key = None
        if not key:
            key = s
        out.append(ids.setdefault(key, len(ids)))
    return np.array(out)


def spearman(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles).astype(np.float64)
    Xt = featurize(test_smiles).astype(np.float64)
    D = descriptors(train_smiles)
    Dt = descriptors(test_smiles)
    keep = D.std(axis=0) > 1e-9
    D, Dt = D[:, keep], Dt[:, keep]
    dmu, dsd = D.mean(axis=0), D.std(axis=0) + 1e-9
    D = np.clip((D - dmu) / dsd, -5.0, 5.0)
    Dt = np.clip((Dt - dmu) / dsd, -5.0, 5.0)
    gamma = 1.0 / D.shape[1]
    T_train = tanimoto(X, X)
    R_train = rbf(D, D, gamma)
    T_test = tanimoto(Xt, X)
    R_test = rbf(Dt, D, gamma)
    y = np.asarray(train_y, dtype=np.float64)
    y = (rankdata(y) - 0.5) / len(y)
    mu, sd = y.mean(), y.std() + 1e-9
    yz = (y - mu) / sd
    groups = scaffold_groups(train_smiles)
    best_score, best_C, best_w = -np.inf, 5.0, 0.5
    for C in (1.0, 5.0, 20.0):
        for w in (0.3, 0.5, 0.7):
            K = w * T_train + (1.0 - w) * R_train
            oof = np.zeros(len(yz))
            for tr, va in GroupKFold(n_splits=5).split(K, yz, groups):
                m = SVR(kernel="precomputed", C=C, epsilon=0.05)
                m.fit(K[np.ix_(tr, tr)], yz[tr])
                oof[va] = m.predict(K[np.ix_(va, tr)])
            score = spearman(oof, y)
            if score > best_score:
                best_score, best_C, best_w = score, C, w
    K = best_w * T_train + (1.0 - best_w) * R_train
    Kt = best_w * T_test + (1.0 - best_w) * R_test
    model = SVR(kernel="precomputed", C=best_C, epsilon=0.05)
    model.fit(K, yz)
    return model.predict(Kt)
