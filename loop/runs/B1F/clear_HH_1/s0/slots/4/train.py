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
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

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
    y = np.asarray(train_y, dtype=np.float64)
    y = (rankdata(y) - 0.5) / len(y)
    mu, sd = y.mean(), y.std() + 1e-9
    yz = (y - mu) / sd
    T_train = tanimoto(X, X)
    T_test = tanimoto(Xt, X)
    # median heuristic on a subsample of training descriptor rows
    rng = np.random.RandomState(seed)
    sub = D[rng.permutation(len(D))[:400]]
    sq = (sub ** 2).sum(axis=1)[:, None] + (sub ** 2).sum(axis=1)[None, :] - 2.0 * sub @ sub.T
    med = np.median(sq[np.triu_indices(len(sub), k=1)])
    base = 1.0 / max(med, 1e-9)
    # inner CV sweep of the gamma multiplier
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_mult, best_score = 1.0, -np.inf
    for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
        Kfull = 0.5 * (T_train + rbf(D, D, base * mult))
        scores = []
        for tr, va in kf.split(D):
            m = SVR(kernel="precomputed", C=5.0, epsilon=0.05)
            m.fit(Kfull[np.ix_(tr, tr)], yz[tr])
            p = m.predict(Kfull[np.ix_(va, tr)])
            scores.append(spearmanr(p, y[va]).correlation)
        s = np.nanmean(scores)
        if s > best_score:
            best_score, best_mult = s, mult
    gamma = base * best_mult
    K_train = 0.5 * (T_train + rbf(D, D, gamma))
    K_test = 0.5 * (T_test + rbf(Dt, D, gamma))
    model = SVR(kernel="precomputed", C=5.0, epsilon=0.05)
    model.fit(K_train, yz)
    return model.predict(K_test)
