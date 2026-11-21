"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge

KRR_ALPHA = 0.3


def tanimoto(A, B):
    A = (A > 0).astype(np.float64)
    B = (B > 0).astype(np.float64)
    inter = A @ B.T
    denom = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return inter / np.maximum(denom, 1e-12)

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


_CACHE = {}


def featurize_one(smi):
    if smi not in _CACHE:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
        m = Chem.MolFromSmiles(smi)
        fp = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6).astype(np.float32)
        _CACHE[smi] = np.concatenate([fp, desc])
    return _CACHE[smi]


def featurize(smiles):
    return np.array([featurize_one(s) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr = featurize(train_smiles)[:, :N_BITS]
    Xte = featurize(test_smiles)[:, :N_BITS]
    y = np.asarray(train_y, dtype=np.float64)
    mu = y.mean()
    model = KernelRidge(kernel='precomputed', alpha=KRR_ALPHA)
    model.fit(tanimoto(Xtr, Xtr), y - mu)
    return model.predict(tanimoto(Xte, Xtr)) + mu
