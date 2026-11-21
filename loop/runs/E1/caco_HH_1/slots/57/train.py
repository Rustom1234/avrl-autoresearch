"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from rdkit.Chem.Scaffolds import MurckoScaffold
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


def phys_one(smi):
    m = Chem.MolFromSmiles(smi)
    n = max(m.GetNumHeavyAtoms(), 1)
    arom = sum(a.GetIsAromatic() for a in m.GetAtoms()) / n
    vals = [f(m) for f in (Descriptors.TPSA, Descriptors.MolLogP, Descriptors.NumHDonors,
                           Descriptors.NumHAcceptors, Descriptors.MolWt, Descriptors.NumRotatableBonds)]
    return vals + [arom]


def phys(smiles):
    return np.array([phys_one(s) for s in smiles], dtype=np.float64)


def knn_feature(fp_a, fp_b, y_b, k=10):
    A = (fp_a > 0).astype(np.float32)
    B = (fp_b > 0).astype(np.float32)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    S = inter / np.maximum(union, 1e-9)
    idx = np.argsort(-S, axis=1)[:, :k]
    w = np.maximum(np.take_along_axis(S, idx, axis=1), 1e-6)
    return (w * y_b[idx]).sum(1) / w.sum(1)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    ptr = phys(train_smiles)
    pte = phys(test_smiles)
    scaler = StandardScaler().fit(ptr)
    lin = Ridge(alpha=1.0).fit(scaler.transform(ptr), y)
    base_tr = lin.predict(scaler.transform(ptr))
    base_te = lin.predict(scaler.transform(pte))
    Xtr = featurize(train_smiles)
    Xte = featurize(test_smiles)
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in train_smiles]
    _, groups = np.unique(scaf, return_inverse=True)
    oof = np.zeros(len(y))
    for tr_i, va_i in GroupKFold(n_splits=5).split(Xtr, y, groups):
        oof[va_i] = knn_feature(Xtr[va_i, :N_BITS], Xtr[tr_i, :N_BITS], y[tr_i])
    kte = knn_feature(Xte[:, :N_BITS], Xtr[:, :N_BITS], y)
    Xtr = np.column_stack([Xtr, oof]).astype(np.float32)
    Xte = np.column_stack([Xte, kte]).astype(np.float32)
    params = dict(tree_method='hist', max_depth=6, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.3, min_child_weight=3,
                  random_state=seed, n_jobs=8)
    stage1 = XGBRegressor(objective='reg:squarederror', n_estimators=200, **params)
    stage1.fit(Xtr, y - base_tr)
    stage2 = XGBRegressor(objective='reg:absoluteerror', n_estimators=400, **params)
    stage2.fit(Xtr, y - base_tr, xgb_model=stage1.get_booster())
    return base_te + stage2.predict(Xte)
