"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.svm import SVR

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mg = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    morgan = np.array([mg.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    desc = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=np.float64)
    desc = np.nan_to_num(desc, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6).astype(np.float32)
    return np.hstack([morgan, desc])


def morgan_counts(smiles):
    mg = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([mg.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float64)


def tanimoto(A, B):
    dot = A @ B.T
    return dot / (np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - dot + 1e-9)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    Xtr, Xte = morgan_counts(train_smiles), morgan_counts(test_smiles)
    model = SVR(kernel="precomputed", C=5, epsilon=0.1)
    model.fit(tanimoto(Xtr, Xtr), y)
    return model.predict(tanimoto(Xte, Xtr))
