"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from scipy.spatial.distance import cdist
from sklearn.svm import SVR

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 500
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float64)


def minmax_kernel(A, B):
    l1 = cdist(A, B, metric="cityblock")
    s = A.sum(1)[:, None] + B.sum(1)[None, :]
    return (s - l1) / np.maximum(s + l1, 1e-9)


def crippen(smiles):
    return np.array([Descriptors.MolLogP(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float64)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr = featurize(train_smiles)
    Xte = featurize(test_smiles)
    model = SVR(kernel="precomputed", C=5.0, epsilon=0.1)
    model.fit(minmax_kernel(Xtr, Xtr), np.asarray(train_y, dtype=np.float64) - crippen(train_smiles))
    return model.predict(minmax_kernel(Xte, Xtr)) + crippen(test_smiles)
