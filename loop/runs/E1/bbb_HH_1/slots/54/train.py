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
N_BINS = 16
EBM_ROUNDS = 40
EBM_LR = 0.1


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


def ebm_scores(D_train, y, D_test, seed):
    """Additive shape-function model: per-feature quantile-bin scores fitted by cyclic shrunk Newton steps."""
    n_feat = D_train.shape[1]
    bins_tr = np.zeros(D_train.shape, dtype=np.int64)
    bins_te = np.zeros(D_test.shape, dtype=np.int64)
    for j in range(n_feat):
        qs = np.unique(np.quantile(D_train[:, j], np.linspace(0, 1, N_BINS + 1)[1:-1]))
        bins_tr[:, j] = np.searchsorted(qs, D_train[:, j], side="right")
        bins_te[:, j] = np.searchsorted(qs, D_test[:, j], side="right")
    shape = np.zeros((n_feat, N_BINS))
    order = np.random.RandomState(seed).permutation(n_feat)
    base = np.log(y.mean() / (1 - y.mean()))
    F_tr = np.full(len(y), base)
    for _ in range(EBM_ROUNDS):
        for j in order:
            p = 1 / (1 + np.exp(-F_tr))
            g = np.bincount(bins_tr[:, j], weights=y - p, minlength=N_BINS)
            h = np.bincount(bins_tr[:, j], weights=p * (1 - p), minlength=N_BINS)
            step = EBM_LR * g / (h + 1.0)
            shape[j] += step
            F_tr += step[bins_tr[:, j]]
    F_te = base + sum(shape[j][bins_te[:, j]] for j in range(n_feat))
    return 1 / (1 + np.exp(-F_te))


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    D_train = full_descriptors(train_smiles)
    D_test = full_descriptors(test_smiles)
    lo, hi = np.percentile(D_train, [1, 99], axis=0)
    X_train = np.hstack([X_train, np.clip(D_train, lo, hi).astype(np.float32)])
    X_test = np.hstack([X_test, np.clip(D_test, lo, hi).astype(np.float32)])
    y = np.asarray(train_y, dtype=np.int64)
    rf = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                n_jobs=-1, random_state=seed)
    rf.fit(X_train, y)
    gb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, colsample_bytree=0.3,
                        random_state=seed, n_jobs=8, verbose=-1)
    gb.fit(X_train, y)
    ebm = ebm_scores(np.clip(D_train, lo, hi), y, np.clip(D_test, lo, hi), seed)
    return (scaled_rank(rf.predict_proba(X_test)[:, 1]) + scaled_rank(gb.predict_proba(X_test)[:, 1])
            + scaled_rank(ebm)) / 3.0
