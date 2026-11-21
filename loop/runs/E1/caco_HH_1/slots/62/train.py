"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
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


def smiles_ngram_block(train_smiles, test_smiles):
    def grams(s):
        c = Counter()
        for n in (2, 3, 4):
            for i in range(len(s) - n + 1):
                c[s[i:i + n]] += 1
        return c

    tr = [grams(s) for s in train_smiles]
    df = Counter()
    for c in tr:
        df.update(c.keys())
    vocab = {g: j for j, g in enumerate(sorted(g for g, k in df.items() if k >= 5))}
    idf = np.array([np.log((len(tr) + 1) / (df[g] + 1)) + 1.0 for g in vocab], dtype=np.float32)

    def vec(counters):
        X = np.zeros((len(counters), len(vocab)), dtype=np.float32)
        for i, c in enumerate(counters):
            for g, v in c.items():
                j = vocab.get(g)
                if j is not None:
                    X[i, j] = (1.0 + np.log(v)) * idf[j]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-9)

    return vec(tr), vec([grams(s) for s in test_smiles])


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
    Gtr, Gte = smiles_ngram_block(train_smiles, test_smiles)
    Xtr = np.hstack([Xtr, Gtr])
    Xte = np.hstack([Xte, Gte])
    params = dict(tree_method='hist', max_depth=6, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.3, min_child_weight=3,
                  random_state=seed, n_jobs=8)
    stage1 = XGBRegressor(objective='reg:squarederror', n_estimators=200, **params)
    stage1.fit(Xtr, y - base_tr)
    stage2 = XGBRegressor(objective='reg:absoluteerror', n_estimators=400, **params)
    stage2.fit(Xtr, y - base_tr, xgb_model=stage1.get_booster())
    return base_te + stage2.predict(Xte)
