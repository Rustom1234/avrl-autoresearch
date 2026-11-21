"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from scipy.stats import rankdata
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
import lightgbm as lgb

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1
KNN = 7
GAMMA_L = 0.5


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
    n = len(train_smiles)
    XA = np.vstack([X, Xt])
    DA = np.vstack([D, Dt])
    K = 0.5 * (tanimoto(XA, XA) + rbf(DA, DA, gamma))
    N = K.shape[0]
    np.fill_diagonal(K, 0.0)
    idx = np.argsort(-K, axis=1)[:, :KNN]
    rows = np.repeat(np.arange(N), KNN)
    W = np.zeros_like(K)
    W[rows, idx.ravel()] = K[rows, idx.ravel()]
    W = np.maximum(W, W.T)
    dinv = 1.0 / np.sqrt(W.sum(axis=1) + 1e-9)
    L = np.eye(N) - dinv[:, None] * W * dinv[None, :]
    y = np.asarray(train_y, dtype=np.float64)
    y = (rankdata(y) - 0.5) / len(y)
    y = (y - y.mean()) / (y.std() + 1e-9)
    s = np.zeros(N)
    s[:n] = 1.0
    yext = np.zeros(N)
    yext[:n] = y
    f = np.linalg.solve(np.diag(s) + GAMMA_L * L, s * yext)
    return f[n:]
