"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, MACCSkeys
from lightgbm import LGBMRegressor

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
    maccs = np.array([list(MACCSkeys.GenMACCSKeys(m)) for m in mols], dtype=np.float32)
    return np.hstack([morgan, desc, maccs])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    model = LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=31, colsample_bytree=0.3,
                          subsample=0.8, subsample_freq=1, min_child_samples=10, objective="l1",
                          n_jobs=8, random_state=seed, verbose=-1)
    model.fit(Xtr, y)
    return model.predict(Xte)
