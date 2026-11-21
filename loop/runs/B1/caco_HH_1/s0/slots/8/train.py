"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
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


ROUND_GRID = [100, 200, 400, 600, 800]
N_STAGE1 = 200


def _scaffold(smi):
    s = MurckoScaffold.MurckoScaffoldSmiles(smiles=smi)
    return s if s else smi


def _two_stage(p_tr, X_tr, y_tr, p_te, X_te, seed, rounds):
    # Ridge base, then stage-1 squared error and stage-2 L1 residual boosting.
    # Returns {n_rounds: prediction} for each stage-2 round count in `rounds`.
    scaler = StandardScaler().fit(p_tr)
    lin = Ridge(alpha=1.0).fit(scaler.transform(p_tr), y_tr)
    base_tr = lin.predict(scaler.transform(p_tr))
    base_te = lin.predict(scaler.transform(p_te))
    params = dict(tree_method='hist', max_depth=6, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.3, min_child_weight=3,
                  random_state=seed, n_jobs=8)
    stage1 = XGBRegressor(objective='reg:squarederror', n_estimators=N_STAGE1, **params)
    stage1.fit(X_tr, y_tr - base_tr)
    stage2 = XGBRegressor(objective='reg:absoluteerror', n_estimators=max(rounds), **params)
    stage2.fit(X_tr, y_tr - base_tr, xgb_model=stage1.get_booster())
    return {r: base_te + stage2.predict(X_te, iteration_range=(0, N_STAGE1 + r)) for r in rounds}


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    ptr = phys(train_smiles)
    pte = phys(test_smiles)
    Xtr = featurize(train_smiles)
    Xte = featurize(test_smiles)
    groups = [_scaffold(s) for s in train_smiles]
    errs = {r: [] for r in ROUND_GRID}
    for tr, va in GroupKFold(n_splits=4).split(Xtr, y, groups):
        preds = _two_stage(ptr[tr], Xtr[tr], y[tr], ptr[va], Xtr[va], seed, ROUND_GRID)
        for r in ROUND_GRID:
            errs[r].append(np.mean(np.abs(preds[r] - y[va])))
    best = min(ROUND_GRID, key=lambda r: np.mean(errs[r]))
    return _two_stage(ptr, Xtr, y, pte, Xte, seed, [best])[best]
