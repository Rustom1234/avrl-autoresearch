"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.svm import SVR

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append(gen.GetCountFingerprintAsNumPy(m).astype(np.float32))
    return np.array(rows)


def minmax_kernel(A, B):
    K = np.zeros((len(A), len(B)))
    for j, b in enumerate(B):
        K[:, j] = np.minimum(A, b).sum(1) / np.maximum(np.maximum(A, b).sum(1), 1e-9)
    return K


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr = featurize(train_smiles)
    Xte = featurize(test_smiles)
    model = SVR(kernel='precomputed', C=5)
    model.fit(minmax_kernel(Xtr, Xtr), np.asarray(train_y, dtype=np.float64))
    return model.predict(minmax_kernel(Xte, Xtr).T if False else minmax_kernel(Xtr, Xte).T)
