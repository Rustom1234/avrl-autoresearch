"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors, Descriptors, Crippen
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def descriptors(mol):
    charge = sum(abs(a.GetFormalCharge()) for a in mol.GetAtoms())
    return [Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumRings(mol), mol.GetNumHeavyAtoms(),
            rdMolDescriptors.CalcFractionCSP3(mol), charge]


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        rows.append(np.concatenate([gen.GetFingerprintAsNumPy(mol), np.array(descriptors(mol), dtype=np.float32)]))
    return np.array(rows, dtype=np.float32)


_DESC_CACHE = {}


def full_descriptors(smiles_list):
    out = []
    for s in smiles_list:
        if s not in _DESC_CACHE:
            mol = Chem.MolFromSmiles(s)
            _DESC_CACHE[s] = [f(mol) for _, f in Descriptors.descList]
        out.append(_DESC_CACHE[s])
    arr = np.array(out, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def scaled_rank(p):
    return (rankdata(p) - 1) / max(len(p) - 1, 1)


def confident_idx(p, frac=0.1):
    # top and bottom `frac` of test molecules by rank, with pseudo-labels 1 / 0
    r = scaled_rank(p)
    hi_idx = np.where(r >= 1 - frac)[0]
    lo_idx = np.where(r <= frac)[0]
    idx = np.concatenate([hi_idx, lo_idx])
    lab = np.concatenate([np.ones(len(hi_idx), dtype=np.int64), np.zeros(len(lo_idx), dtype=np.int64)])
    return idx, lab


def fit_view_fp(X, y, seed):
    m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, colsample_bytree=0.3,
                       random_state=seed, n_jobs=8, verbose=-1)
    return m.fit(X, y)


def fit_view_desc(X, y, seed):
    m = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                               n_jobs=-1, random_state=seed)
    return m.fit(X, y)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xa_train = featurize(train_smiles)[:, :N_BITS]
    Xa_test = featurize(test_smiles)[:, :N_BITS]
    D_train = full_descriptors(train_smiles)
    D_test = full_descriptors(test_smiles)
    lo, hi = np.percentile(D_train, [1, 99], axis=0)
    Xb_train = np.clip(D_train, lo, hi).astype(np.float32)
    Xb_test = np.clip(D_test, lo, hi).astype(np.float32)
    y = np.asarray(train_y, dtype=np.int64)

    ma = fit_view_fp(Xa_train, y, seed)
    mb = fit_view_desc(Xb_train, y, seed)
    for _ in range(2):
        pa = ma.predict_proba(Xa_test)[:, 1]
        pb = mb.predict_proba(Xb_test)[:, 1]
        ia, la = confident_idx(pa)
        ib, lb = confident_idx(pb)
        ma = fit_view_fp(np.vstack([Xa_train, Xa_test[ib]]), np.concatenate([y, lb]), seed)
        mb = fit_view_desc(np.vstack([Xb_train, Xb_test[ia]]), np.concatenate([y, la]), seed)
    return 0.5 * (scaled_rank(ma.predict_proba(Xa_test)[:, 1]) + scaled_rank(mb.predict_proba(Xb_test)[:, 1]))
