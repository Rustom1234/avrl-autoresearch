"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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
    r = y - base_tr
    # inner scaffold-grouped validation fold (1 of 4) to choose stage-2 rounds
    scaf = np.array([MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in train_smiles])
    uniq = np.unique(scaf)
    order = np.random.RandomState(seed).permutation(len(uniq))
    fold_of = {uniq[j]: i % 4 for i, j in enumerate(order)}
    val = np.array([fold_of[s] == 0 for s in scaf])
    tr = ~val
    s1 = XGBRegressor(objective='reg:squarederror', n_estimators=200, **params)
    s1.fit(Xtr[tr], r[tr])
    s2 = XGBRegressor(objective='reg:absoluteerror', n_estimators=1000, eval_metric='mae',
                      early_stopping_rounds=50, **params)
    s2.fit(Xtr[tr], r[tr], eval_set=[(Xtr[val], r[val])], xgb_model=s1.get_booster(), verbose=False)
    hist = s2.evals_result()['validation_0']['mae']
    n2 = int(np.argmin(hist)) + 1
    stage1 = XGBRegressor(objective='reg:squarederror', n_estimators=200, **params)
    stage1.fit(Xtr, r)
    stage2 = XGBRegressor(objective='reg:absoluteerror', n_estimators=n2, **params)
    stage2.fit(Xtr, r, xgb_model=stage1.get_booster())
    return base_te + stage2.predict(Xte)
