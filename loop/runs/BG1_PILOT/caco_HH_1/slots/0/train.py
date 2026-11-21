"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge

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
    dot = A @ B.T
    na = (A * A).sum(1)[:, None]
    nb = (B * B).sum(1)[None, :]
    return dot / (na + nb - dot + 1e-9)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr = featurize(train_smiles).astype(np.float64)
    Xte = featurize(test_smiles).astype(np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    mu = y.mean()
    model = KernelRidge(kernel='precomputed', alpha=0.5)
    model.fit(tanimoto(Xtr, Xtr), y - mu)
    return model.predict(tanimoto(Xte, Xtr)) + mu
