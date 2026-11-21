"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors, Descriptors, Crippen
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from rdkit import DataStructs

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


def morgan_fps(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def tanimoto_knn(train_fps, train_y, query_fps, k=10):
    out = np.zeros(len(query_fps))
    for i, fp in enumerate(query_fps):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        idx = np.argsort(-sims)[:k]
        w = sims[idx] ** 3
        out[i] = np.sum(w * train_y[idx]) / max(np.sum(w), 1e-9)
    return out


def fit_base(X, y, fps, seed):
    rf = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                n_jobs=-1, random_state=seed)
    rf.fit(X, y)
    gb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, colsample_bytree=0.3,
                        random_state=seed, n_jobs=8, verbose=-1)
    gb.fit(X, y)

    def predict(Xq, fq):
        return np.column_stack([rf.predict_proba(Xq)[:, 1], gb.predict_proba(Xq)[:, 1],
                                tanimoto_knn(fps, y, fq)])
    return predict


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    D_train = full_descriptors(train_smiles)
    D_test = full_descriptors(test_smiles)
    lo, hi = np.percentile(D_train, [1, 99], axis=0)
    X_train = np.hstack([X_train, np.clip(D_train, lo, hi).astype(np.float32)])
    X_test = np.hstack([X_test, np.clip(D_test, lo, hi).astype(np.float32)])
    y = np.asarray(train_y, dtype=np.int64)
    fps_train = morgan_fps(train_smiles)
    fps_test = morgan_fps(test_smiles)
    oof = np.zeros((len(y), 3))
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train, y):
        model = fit_base(X_train[tr], y[tr], [fps_train[i] for i in tr], seed)
        oof[te] = model(X_train[te], [fps_train[i] for i in te])
    full = fit_base(X_train, y, fps_train, seed)
    meta = LogisticRegression(C=1.0, max_iter=1000).fit(oof, y)
    return meta.predict_proba(full(X_test, fps_test))[:, 1]
