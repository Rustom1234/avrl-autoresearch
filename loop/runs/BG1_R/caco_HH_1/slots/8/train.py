"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


_CACHE = {}
_PCACHE = {}


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


def featurize_pharm_one(smi):
    if smi not in _PCACHE:
        m = Chem.MolFromSmiles(smi)
        pf = Generate.Gen2DFingerprint(m, Gobbi_Pharm2D.factory)
        arr = np.zeros(pf.GetNumBits(), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(pf, arr)
        _PCACHE[smi] = arr
    return _PCACHE[smi]


def featurize_pharm(smiles):
    return np.array([featurize_pharm_one(s) for s in smiles], dtype=np.uint8)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = XGBRegressor(objective='reg:absoluteerror', tree_method='hist', max_depth=6, learning_rate=0.03,
                         n_estimators=600, subsample=0.8, colsample_bytree=0.3, min_child_weight=3,
                         random_state=seed, n_jobs=8)
    Ptr = featurize_pharm(train_smiles)
    keep = Ptr.mean(axis=0) >= 0.02
    Xtr = np.hstack([featurize(train_smiles), Ptr[:, keep]])
    Xte = np.hstack([featurize(test_smiles), featurize_pharm(test_smiles)[:, keep]])
    model.fit(Xtr, np.asarray(train_y, dtype=np.float64))
    return model.predict(Xte)
