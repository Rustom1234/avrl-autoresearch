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


ALPHA_LP = 0.9
LP_ITERS = 30
LP_K = 10
LP_WEIGHT = 0.2


def label_spread(fp_train, y_train, fp_test):
    F_all = np.vstack([fp_train, fp_test]).astype(np.float64)
    inter = F_all @ F_all.T
    cnt = F_all.sum(axis=1)
    sim = inter / np.maximum(cnt[:, None] + cnt[None, :] - inter, 1e-9)
    np.fill_diagonal(sim, 0.0)
    n = sim.shape[0]
    idx = np.argsort(-sim, axis=1)[:, :LP_K]
    rows = np.repeat(np.arange(n), LP_K)
    W = np.zeros_like(sim)
    W[rows, idx.ravel()] = sim[rows, idx.ravel()]
    W = np.maximum(W, W.T)
    d = W.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    S = dinv[:, None] * W * dinv[None, :]
    n_tr = len(y_train)
    Y0 = np.zeros(n)
    Y0[:n_tr] = 2.0 * np.asarray(y_train, dtype=np.float64) - 1.0
    F = Y0.copy()
    for _ in range(LP_ITERS):
        F = ALPHA_LP * (S @ F) + (1 - ALPHA_LP) * Y0
    return F[n_tr:]


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    D_train = full_descriptors(train_smiles)
    D_test = full_descriptors(test_smiles)
    lo, hi = np.percentile(D_train, [1, 99], axis=0)
    X_train = np.hstack([X_train, np.clip(D_train, lo, hi).astype(np.float32)])
    X_test = np.hstack([X_test, np.clip(D_test, lo, hi).astype(np.float32)])
    graph = label_spread(X_train[:, :N_BITS], train_y, X_test[:, :N_BITS])
    y = np.asarray(train_y, dtype=np.int64)
    rf = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                n_jobs=-1, random_state=seed)
    rf.fit(X_train, y)
    gb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, colsample_bytree=0.3,
                        random_state=seed, n_jobs=8, verbose=-1)
    gb.fit(X_train, y)
    tree = 0.5 * (scaled_rank(rf.predict_proba(X_test)[:, 1]) + scaled_rank(gb.predict_proba(X_test)[:, 1]))
    return (1 - LP_WEIGHT) * tree + LP_WEIGHT * scaled_rank(graph)
