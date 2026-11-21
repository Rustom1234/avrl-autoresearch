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
from sklearn.linear_model import ARDRegression

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
    gamma = 1.0 / D.shape[1]
    K_train = 0.5 * (tanimoto(X, X) + rbf(D, D, gamma))
    K_test = 0.5 * (tanimoto(Xt, X) + rbf(Dt, D, gamma))
    y = np.asarray(train_y, dtype=np.float64)
    y = (rankdata(y) - 0.5) / len(y)
    mu, sd = y.mean(), y.std() + 1e-9
    ys = (y - mu) / sd
    n = len(ys)
    lam, delta = 0.3, 0.2
    w = np.ones(n)
    for _ in range(10):
        alpha = np.linalg.solve(w[:, None] * K_train + lam * np.eye(n), w * ys)
        r = ys - K_train @ alpha
        w = np.minimum(1.0, delta / np.maximum(np.abs(r), 1e-12))
    svr_pred = K_test @ alpha
    ard = ARDRegression(max_iter=300)
    ard.fit(D, y)
    ard_pred = ard.predict(Dt)
    return 0.7 * rankdata(svr_pred) / len(svr_pred) + 0.3 * rankdata(ard_pred) / len(ard_pred)
