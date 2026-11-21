"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors, Descriptors, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1
HARD_WEIGHT = 2.0


def scaffold_groups(smiles):
    ids = {}
    out = []
    for s in smiles:
        key = MurckoScaffold.MurckoScaffoldSmiles(smiles=s)
        out.append(ids.setdefault(key, len(ids)))
    return np.array(out)


def fit_predict_pair(X_fit, y_fit, X_eval, w, seed):
    rf = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                n_jobs=-1, random_state=seed)
    rf.fit(X_fit, y_fit, sample_weight=w)
    gb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, colsample_bytree=0.3,
                        random_state=seed, n_jobs=8, verbose=-1)
    gb.fit(X_fit, y_fit, sample_weight=w)
    return rf.predict_proba(X_eval)[:, 1], gb.predict_proba(X_eval)[:, 1]


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


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    D_train = full_descriptors(train_smiles)
    D_test = full_descriptors(test_smiles)
    lo, hi = np.percentile(D_train, [1, 99], axis=0)
    X_train = np.hstack([X_train, np.clip(D_train, lo, hi).astype(np.float32)])
    X_test = np.hstack([X_test, np.clip(D_test, lo, hi).astype(np.float32)])
    y = np.asarray(train_y, dtype=np.int64)
    # First pass: out-of-fold scores, grouped by scaffold, with a seed-dependent group shuffle.
    rng = np.random.RandomState(seed)
    groups = scaffold_groups(train_smiles)
    groups = rng.permutation(groups.max() + 1)[groups]
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X_train, y, groups):
        p_rf, p_gb = fit_predict_pair(X_train[tr], y[tr], X_train[te], np.ones(len(tr)), seed)
        oof[te] = 0.5 * (p_rf + p_gb)
    oof_rank = scaled_rank(oof)
    # Hard examples: positives ranked in the bottom half or negatives ranked in the top half.
    wrong = ((y == 1) & (oof_rank < 0.5)) | ((y == 0) & (oof_rank > 0.5))
    w = np.where(wrong, HARD_WEIGHT, 1.0)
    # Second pass: refit both members with the hard-example weights and rank-average as before.
    p_rf, p_gb = fit_predict_pair(X_train, y, X_test, w, seed)
    return 0.5 * (scaled_rank(p_rf) + scaled_rank(p_gb))
