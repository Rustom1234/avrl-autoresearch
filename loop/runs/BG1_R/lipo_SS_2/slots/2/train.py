"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.linalg import solve

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 500
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


NOISE = 0.3


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def minmax_kernel(A, B):
    # sum(min) = sum over thresholds t of <1[A>=t], 1[B>=t>; sum(max) = |A| + |B| - sum(min)
    mn = np.zeros((len(A), len(B)), dtype=np.float32)
    for t in range(1, int(max(A.max(), B.max())) + 1):
        mn += (A >= t).astype(np.float32) @ (B >= t).astype(np.float32).T
    mx = A.sum(1)[:, None] + B.sum(1)[None, :] - mn
    return (mn / np.maximum(mx, 1e-9)).astype(np.float64)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    Xt = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    mu = y.mean()
    K = minmax_kernel(X, X) + NOISE * np.eye(len(X))
    alpha = solve(K, y - mu, assume_a="pos")
    return minmax_kernel(Xt, X) @ alpha + mu
