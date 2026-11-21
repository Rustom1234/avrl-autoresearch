"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from scipy.stats import rankdata
from collections import Counter
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fp = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        rows.append(np.concatenate([desc, fp]))
    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


N_SCAF = 512


def n_ring_systems(mol):
    systems = []
    for ring in mol.GetRingInfo().AtomRings():
        r = set(ring)
        merged = [s for s in systems if s & r]
        for s in merged:
            systems.remove(s)
            r = r | s
        systems.append(r)
    return len(systems)


def scaffold_info(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        core = MurckoScaffold.GetScaffoldForMol(m)
        gen = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core))
        out.append((gen, n_ring_systems(core), core.GetRingInfo().NumRings()))
    return out


def scaffold_matrix(info, vocab):
    idx = {s: i for i, s in enumerate(vocab)}
    M = np.zeros((len(info), len(vocab) + 2), dtype=np.float32)
    for r, (g, nsys, nring) in enumerate(info):
        if g in idx:
            M[r, idx[g]] = 1.0
        M[r, -2] = nsys
        M[r, -1] = nring
    return M


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    Xt = featurize(test_smiles)
    tr_info = scaffold_info(train_smiles)
    te_info = scaffold_info(test_smiles)
    vocab = [s for s, _ in Counter(g for g, _, _ in tr_info).most_common(N_SCAF)]
    X = np.hstack([X, scaffold_matrix(tr_info, vocab)])
    Xt = np.hstack([Xt, scaffold_matrix(te_info, vocab)])
    y = np.asarray(train_y, dtype=np.int64)
    models = [
        RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                               n_jobs=-1, random_state=seed),
        LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.3, n_jobs=8, random_state=seed, verbose=-1),
        CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, rsm=0.2, thread_count=8,
                           random_seed=seed, verbose=0),
    ]
    preds = []
    for m in models:
        m.fit(X, y)
        preds.append(rankdata(m.predict_proba(Xt)[:, 1]) / len(Xt))
    return np.mean(preds, axis=0)
