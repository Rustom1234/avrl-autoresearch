"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

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
    model = XGBRegressor(objective='reg:absoluteerror', tree_method='hist', max_depth=6, learning_rate=0.03,
                         n_estimators=600, subsample=0.8, colsample_bytree=0.3, min_child_weight=3,
                         random_state=seed, n_jobs=8)
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in train_smiles]
    counts = Counter(scaffolds)
    weights = np.array([1.0 / np.sqrt(counts[sc]) for sc in scaffolds], dtype=np.float64)
    weights = weights / weights.mean()
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.float64), sample_weight=weights)
    return model.predict(featurize(test_smiles))
