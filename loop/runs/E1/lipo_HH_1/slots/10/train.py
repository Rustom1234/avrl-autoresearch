"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
import lightgbm as lgb
from catboost import CatBoostRegressor
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# Settings
N_BITS = 1024
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    ecfp = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fcfp = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, fpSize=N_BITS,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        rows.append(np.concatenate([ecfp.GetFingerprintAsNumPy(mol), fcfp.GetFingerprintAsNumPy(mol)]))
    return np.array(rows, dtype=np.float32)


_DESC_CACHE = {}


def desc_vector(smiles):
    # Cache per SMILES: descriptor computation is slow and the same molecules recur across the 15 fits.
    if smiles not in _DESC_CACHE:
        mol = Chem.MolFromSmiles(smiles)
        _DESC_CACHE[smiles] = np.array([f(mol) for _, f in Descriptors._descList], dtype=np.float64)
    return _DESC_CACHE[smiles]


N_CONFIGS = 1
N_ROUNDS = 600
LGB_GRID = [(nl, mcs, ff) for nl in (15, 31, 63) for mcs in (5, 10, 20) for ff in (0.2, 0.4)]


def lgb_params(cfg, seed):
    nl, mcs, ff = cfg
    return dict(objective="regression_l1", learning_rate=0.03, num_leaves=nl, min_child_samples=mcs,
                feature_fraction=ff, seed=seed, verbose=-1, num_threads=8)


def tanimoto(A, B):
    dot = A @ B.T
    na = (A * A).sum(axis=1)[:, None]
    nb = (B * B).sum(axis=1)[None, :]
    return dot / np.maximum(na + nb - dot, 1e-9)


def base_predict(Xa, Fa, ya, Xb, Fb, seed):
    # Three base learners; returns their predictions on Xb as columns.
    cb = CatBoostRegressor(loss_function="MAE", iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                           random_seed=seed, verbose=False, thread_count=8, border_count=64)
    p_rf = cb.fit(Xa, ya).predict(Xb)
    m = lgb.train(lgb_params((31, 10, 0.2), seed), lgb.Dataset(Xa, ya), num_boost_round=N_ROUNDS)
    p_lgb = m.predict(Xb)
    mu = ya.mean()
    w = np.linalg.solve(tanimoto(Fa, Fa) + np.eye(len(ya)), ya - mu)
    p_krr = tanimoto(Fb, Fa) @ w + mu
    return np.column_stack([p_rf, p_lgb, p_krr])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Dtr = np.nan_to_num(np.array([desc_vector(s) for s in train_smiles]), nan=0.0, posinf=0.0, neginf=0.0)
    Dte = np.nan_to_num(np.array([desc_vector(s) for s in test_smiles]), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(Dtr, 1, axis=0), np.percentile(Dtr, 99, axis=0)
    Ftr, Fte = featurize(train_smiles), featurize(test_smiles)
    Xtr = np.hstack([Ftr, np.clip(Dtr, lo, hi).astype(np.float32)])
    Xte = np.hstack([Fte, np.clip(Dte, lo, hi).astype(np.float32)])
    # Tanimoto kernel uses the ECFP bits only (first N_BITS columns).
    Ftr, Fte = Ftr[:, :N_BITS].astype(np.float64), Fte[:, :N_BITS].astype(np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    groups = np.array([MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in train_smiles])
    oof = np.zeros((len(y), 3))
    for tr, va in GroupKFold(n_splits=5).split(Xtr, y, groups):
        oof[va] = base_predict(Xtr[tr], Ftr[tr], y[tr], Xtr[va], Ftr[va], seed)
    meta = Ridge(alpha=1e-3, positive=True).fit(oof, y)
    base_te = base_predict(Xtr, Ftr, y, Xte, Fte, seed)
    return meta.predict(base_te)
