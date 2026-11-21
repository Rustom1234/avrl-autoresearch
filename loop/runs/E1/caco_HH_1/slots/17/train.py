"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Settings
RADIUS = 2
N_VIEWS = 4
VIEW_BITS = 512
VIEW_MULTS = [1103515245, 69069, 22695477, 1664525]
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


_CACHE = {}


def featurize_one(smi):
    if smi not in _CACHE:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS)
        m = Chem.MolFromSmiles(smi)
        counts = gen.GetSparseCountFingerprint(m).GetNonzeroElements()
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6).astype(np.float32)
        _CACHE[smi] = (counts, desc)
    return _CACHE[smi]


def featurize_view(smiles, v):
    # Salted hash of the unfolded Morgan IDs into VIEW_BITS buckets; each view has its own collision pattern.
    mult = VIEW_MULTS[v]
    salt = 40503 * (v + 1)
    rows = []
    for s in smiles:
        counts, desc = featurize_one(s)
        x = np.zeros(VIEW_BITS + desc.shape[0], dtype=np.float32)
        for fid, c in counts.items():
            x[((fid * mult + salt) % 2147483647) % VIEW_BITS] += c
        x[VIEW_BITS:] = desc
        rows.append(x)
    return np.array(rows, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    preds = []
    for v in range(N_VIEWS):
        model = XGBRegressor(objective='reg:absoluteerror', tree_method='hist', max_depth=6, learning_rate=0.03,
                             n_estimators=600, subsample=0.8, colsample_bytree=0.3, min_child_weight=3,
                             random_state=seed * N_VIEWS + v, n_jobs=8)
        model.fit(featurize_view(train_smiles, v), y)
        preds.append(model.predict(featurize_view(test_smiles, v)))
    return np.mean(preds, axis=0)
