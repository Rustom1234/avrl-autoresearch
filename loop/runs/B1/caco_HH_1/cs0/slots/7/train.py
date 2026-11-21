"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from rdkit.Chem.Scaffolds import MurckoScaffold

GRID = [100, 200, 400, 600, 800]

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


def _fit(Xa, pa, ya, rounds, seed):
    scaler = StandardScaler().fit(pa)
    lin = Ridge(alpha=1.0).fit(scaler.transform(pa), ya)
    base = lin.predict(scaler.transform(pa))
    params = dict(tree_method='hist', max_depth=6, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.3, min_child_weight=3,
                  random_state=seed, n_jobs=8)
    stage1 = XGBRegressor(objective='reg:squarederror', n_estimators=200, **params)
    stage1.fit(Xa, ya - base)
    stage2 = XGBRegressor(objective='reg:absoluteerror', n_estimators=rounds, **params)
    stage2.fit(Xa, ya - base, xgb_model=stage1.get_booster())
    return scaler, lin, stage2


def _predict(model, X, p, k):
    scaler, lin, stage2 = model
    return lin.predict(scaler.transform(p)) + stage2.predict(X, iteration_range=(0, 200 + k))


def _scaffold(smi):
    try:
        s = MurckoScaffold.MurckoScaffoldSmiles(smiles=smi)
    except Exception:
        s = ''
    return s if s else smi


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    ptr = phys(train_smiles)
    pte = phys(test_smiles)
    Xtr = featurize(train_smiles)
    Xte = featurize(test_smiles)
    groups = np.array([_scaffold(s) for s in train_smiles])
    curve = np.zeros(len(GRID))
    n_splits = 4
    for a, b in GroupKFold(n_splits=n_splits).split(Xtr, y, groups):
        model = _fit(Xtr[a], ptr[a], y[a], max(GRID), seed)
        for i, k in enumerate(GRID):
            curve[i] += np.mean(np.abs(_predict(model, Xtr[b], ptr[b], k) - y[b])) / n_splits
    best = GRID[int(np.argmin(curve))]
    model = _fit(Xtr, ptr, y, best, seed)
    return _predict(model, Xte, pte, best)
