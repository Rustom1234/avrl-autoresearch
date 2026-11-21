"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from lightgbm import LGBMRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
FEATURE_FRACTION = 0.8
BAG_FRACTION = 0.8


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = LGBMRegressor(n_estimators=N_TREES, 
                             feature_fraction=FEATURE_FRACTION, 
                             bag_fraction=BAG_FRACTION, 
                             random_state=seed, 
                             n_jobs=-1, 
                             verbosity=-1)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles))
