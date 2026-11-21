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


def describe(smiles):
    rows = [[f(Chem.MolFromSmiles(s)) for _, f in Descriptors._descList] for s in smiles]
    d = np.array(rows, dtype=np.float64)
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Dtr = describe(train_smiles)
    Dte = describe(test_smiles)
    lo, hi = np.percentile(Dtr, [0.5, 99.5], axis=0)
    Xtr = np.hstack([featurize(train_smiles), np.clip(Dtr, lo, hi)])
    Xte = np.hstack([featurize(test_smiles), np.clip(Dte, lo, hi)])
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    model.fit(Xtr, np.asarray(train_y, dtype=np.float64))
    return model.predict(Xte)
