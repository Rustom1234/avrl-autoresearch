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
    params = dict(tree_method='hist', max_depth=6, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.3, min_child_weight=3,
                  random_state=seed, n_jobs=8)

    def fit_two(X, t, n2):
        s1 = XGBRegressor(objective='reg:squarederror', n_estimators=200, **params)
        s1.fit(X, t)
        s2 = XGBRegressor(objective='reg:absoluteerror', n_estimators=n2, **params)
        s2.fit(X, t, xgb_model=s1.get_booster())
        return s2

    grid = [100, 200, 400, 700, 1000]
    groups = []
    for s in train_smiles:
        try:
            groups.append(MurckoScaffold.MurckoScaffoldSmiles(smi=s))
        except Exception:
            groups.append(s)
    groups = np.array(groups)
    resid = y - base_tr
    curves = np.zeros(len(grid))
    for tr_i, va_i in GroupKFold(n_splits=4).split(Xtr, y, groups):
        m = fit_two(Xtr[tr_i], resid[tr_i], max(grid))
        for j, k in enumerate(grid):
            p = base_tr[va_i] + m.predict(Xtr[va_i], iteration_range=(0, 200 + k))
            curves[j] += np.abs(p - y[va_i]).mean()
    best = grid[int(np.argmin(curves))]
    final = fit_two(Xtr, resid, best)
    return base_te + final.predict(Xte)
