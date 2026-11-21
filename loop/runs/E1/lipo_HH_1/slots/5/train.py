"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
import lightgbm as lgb
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold

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


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Dtr = np.nan_to_num(np.array([desc_vector(s) for s in train_smiles]), nan=0.0, posinf=0.0, neginf=0.0)
    Dte = np.nan_to_num(np.array([desc_vector(s) for s in test_smiles]), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(Dtr, 1, axis=0), np.percentile(Dtr, 99, axis=0)
    Xtr = np.hstack([featurize(train_smiles), np.clip(Dtr, lo, hi).astype(np.float32)])
    Xte = np.hstack([featurize(test_smiles), np.clip(Dte, lo, hi).astype(np.float32)])
    y = np.asarray(train_y, dtype=np.float64)
    groups = np.array([MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in train_smiles])
    rng = np.random.RandomState(seed)
    configs = [LGB_GRID[i] for i in rng.choice(len(LGB_GRID), N_CONFIGS, replace=False)]
    best_cfg, best_score = None, np.inf
    for cfg in configs:
        errs = []
        for tr, va in GroupKFold(n_splits=4).split(Xtr, y, groups):
            m = lgb.train(lgb_params(cfg, seed), lgb.Dataset(Xtr[tr], y[tr]), num_boost_round=N_ROUNDS)
            errs.append(np.mean(np.abs(m.predict(Xtr[va]) - y[va])))
        score = np.mean(errs)
        if score < best_score:
            best_score, best_cfg = score, cfg
    model = lgb.train(lgb_params(best_cfg, seed), lgb.Dataset(Xtr, y), num_boost_round=N_ROUNDS)
    return model.predict(Xte)
