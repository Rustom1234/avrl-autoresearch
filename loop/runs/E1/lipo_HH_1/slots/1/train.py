"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


_DESC_CACHE = {}


def desc_vector(smiles):
    # Cache per SMILES: descriptor computation is slow and the same molecules recur across the 15 fits.
    if smiles not in _DESC_CACHE:
        mol = Chem.MolFromSmiles(smiles)
        _DESC_CACHE[smiles] = np.array([f(mol) for _, f in Descriptors._descList], dtype=np.float64)
    return _DESC_CACHE[smiles]


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    Dtr = np.nan_to_num(np.array([desc_vector(s) for s in train_smiles]), nan=0.0, posinf=0.0, neginf=0.0)
    Dte = np.nan_to_num(np.array([desc_vector(s) for s in test_smiles]), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(Dtr, 1, axis=0), np.percentile(Dtr, 99, axis=0)
    Xtr = np.hstack([featurize(train_smiles), np.clip(Dtr, lo, hi).astype(np.float32)])
    Xte = np.hstack([featurize(test_smiles), np.clip(Dte, lo, hi).astype(np.float32)])
    model.fit(Xtr, np.asarray(train_y, dtype=np.float64))
    return model.predict(Xte)
