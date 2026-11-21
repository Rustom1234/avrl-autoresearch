"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
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


N_BINS = 10


def _model_predict(train_smiles, y, test_smiles, seed):
    ptr = phys(train_smiles)
    pte = phys(test_smiles)
    scaler = StandardScaler().fit(ptr)
    lin = Ridge(alpha=1.0).fit(scaler.transform(ptr), y)
    base_tr = lin.predict(scaler.transform(ptr))
    base_te = lin.predict(scaler.transform(pte))
    model = XGBRegressor(objective='reg:absoluteerror', tree_method='hist', max_depth=6, learning_rate=0.03,
                         n_estimators=600, subsample=0.8, colsample_bytree=0.3, min_child_weight=3,
                         random_state=seed, n_jobs=8)
    model.fit(featurize(train_smiles), y - base_tr)
    return base_te + model.predict(featurize(test_smiles))


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    smi = np.asarray(train_smiles)
    groups = np.array([MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in smi])
    oof = np.zeros(len(y))
    for tr_idx, te_idx in GroupKFold(n_splits=5).split(smi, y, groups):
        oof[te_idx] = _model_predict(smi[tr_idx].tolist(), y[tr_idx], smi[te_idx].tolist(), seed)
    resid = y - oof
    edges = np.quantile(oof, np.linspace(0, 1, N_BINS + 1)[1:-1])
    bins = np.searchsorted(edges, oof, side='right')
    med = np.array([np.median(resid[bins == b]) if np.any(bins == b) else 0.0 for b in range(N_BINS)])
    pred = _model_predict(list(train_smiles), y, list(test_smiles), seed)
    tb = np.clip(np.searchsorted(edges, pred, side='right'), 0, N_BINS - 1)
    return pred + med[tb]
