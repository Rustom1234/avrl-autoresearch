"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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
    # RDKit descriptors only (columns after the Morgan block), clipped to training percentiles
    Xtr = featurize(train_smiles)[:, N_BITS:].astype(np.float64)
    Xte = featurize(test_smiles)[:, N_BITS:].astype(np.float64)
    lo, hi = np.percentile(Xtr, 1, axis=0), np.percentile(Xtr, 99, axis=0)
    Xtr = np.clip(Xtr, lo, hi)
    Xte = np.clip(Xte, lo, hi)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)
    y = np.asarray(train_y, dtype=np.float64)
    sd = y.std()
    model = MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-2, learning_rate_init=1e-3,
                         early_stopping=True, validation_fraction=0.15, max_iter=500, random_state=seed)
    model.fit(Xtr, y / sd)
    return model.predict(Xte) * sd
