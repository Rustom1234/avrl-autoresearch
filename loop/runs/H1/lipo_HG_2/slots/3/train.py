"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles, mask=None):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fps = np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)
    if mask is not None:
        return fps[:, mask]
    return fps


def fit_predict(train_smiles, train_y, test_smiles, seed):
    train_fps = featurize(train_smiles)
    # Find bits that are set in at least 2 molecules
    counts = np.sum(train_fps, axis=0)
    mask = counts >= 2
    
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    model.fit(train_fps[:, mask], np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles, mask=mask))
