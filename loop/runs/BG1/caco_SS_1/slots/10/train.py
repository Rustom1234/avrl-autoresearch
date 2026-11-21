"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from lightgbm import LGBMRegressor

# Settings
N_BITS = 1024
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        cnt = gen.GetCountFingerprintAsNumPy(m).astype(np.float64)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        rows.append(desc)
    X = np.nan_to_num(np.array(rows), nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.5, min_child_samples=10,
                          random_state=seed, n_jobs=8, verbose=-1)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles))
