"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter
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


def inner_val_mask(smiles, seed, k=4):
    # Scaffold-grouped k-fold split; returns True for molecules in inner fold 0.
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) for s in smiles]
    counts = Counter(scaf)
    groups = sorted(counts)
    np.random.RandomState(seed).shuffle(groups)
    sizes = np.zeros(k)
    fold_of = {}
    for g in groups:
        f = int(np.argmin(sizes))
        fold_of[g] = f
        sizes[f] += counts[g]
    return np.array([fold_of[s] == 0 for s in scaf])


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
    r = y - base_tr
    params = dict(tree_method='hist', max_depth=6, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.3, min_child_weight=3,
                  random_state=seed, n_jobs=8)
    # Inner split: early-stop stage 2 on one scaffold-grouped inner fold.
    vm = inner_val_mask(train_smiles, seed)
    s1 = XGBRegressor(objective='reg:squarederror', n_estimators=200, **params)
    s1.fit(Xtr[~vm], r[~vm])
    s2 = XGBRegressor(objective='reg:absoluteerror', n_estimators=1000,
                      early_stopping_rounds=50, **params)
    s2.fit(Xtr[~vm], r[~vm], eval_set=[(Xtr[vm], r[vm])],
           xgb_model=s1.get_booster(), verbose=False)
    n2 = s2.best_iteration + 1
    # Refit both stages on all training data.
    f1 = XGBRegressor(objective='reg:squarederror', n_estimators=200, **params)
    f1.fit(Xtr, r)
    f2 = XGBRegressor(objective='reg:absoluteerror', n_estimators=n2, **params)
    f2.fit(Xtr, r, xgb_model=f1.get_booster())
    return base_te + f2.predict(Xte)
