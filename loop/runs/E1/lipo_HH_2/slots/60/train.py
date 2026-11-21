"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# Settings
N_BITS = 2048
RADIUS = 2
N_ESTIMATORS = 800
LEARNING_RATE = 0.05


SMARTS_PATS = [
    "[CX3](=O)[OX2H1,OX1-]",                                   # carboxylic acid
    "c1nnnn1",                                                 # tetrazole
    "S(=O)(=O)[OX2H1,OX1-]",                                   # sulfonic acid
    "C(=O)[NH]S(=O)=O",                                        # acyl sulfonamide
    "[NX3;!$(N-a);!$(N-C=[O,S,N]);!$(N-S)]-[CX4]",             # aliphatic amine (basic)
    "[NX4+]",                                                  # quaternary nitrogen
    "c[OX2H1]",                                                # phenol
    "C(=O)[NX3]",                                              # amide
    "[CX4][OX2H1]",                                            # hydroxyl
    "[N+](=O)[O-]",                                            # nitro
]
SMARTS_MOLS = [Chem.MolFromSmarts(p) for p in SMARTS_PATS]


def smarts_counts(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([len(m.GetSubstructMatches(p)) for p in SMARTS_MOLS])
    return np.array(rows, dtype=np.float32)


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    ap_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
    tt_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fp = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    ap = np.array([ap_gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    tt = np.array([tt_gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    return np.hstack([fp, smarts_counts(smiles), ap, tt])


_DESC_CACHE = {}


def desc_matrix(smiles):
    # RDKit 2D descriptors, cached per SMILES so repeated fits do not recompute them
    missing = [s for s in set(smiles) if s not in _DESC_CACHE]
    for s in missing:
        vals = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s))
        v = np.array(list(vals.values()), dtype=np.float64)
        _DESC_CACHE[s] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return np.array([_DESC_CACHE[s] for s in smiles])


def tanimoto(A, B):
    dot = A @ B.T
    na = (A * A).sum(1)[:, None]
    nb = (B * B).sum(1)[None, :]
    return dot / np.maximum(na + nb - dot, 1e-9)


def huber_krr(K, y, Kte, lam=1.0, delta=0.3, iters=8):
    # Iteratively reweighted kernel ridge with Huber weights on the precomputed kernel K
    from scipy.linalg import cho_factor, cho_solve
    w = np.ones(len(y))
    for _ in range(iters):
        alpha = cho_solve(cho_factor(K + np.diag(lam / w)), y)
        r = y - K @ alpha
        w = np.minimum(1.0, delta / np.maximum(np.abs(r), 1e-9))
    return Kte @ alpha


def base_predict(Xtr, ytr, Xte, seed):
    lg = lgb.LGBMRegressor(objective="l1", n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
                           colsample_bytree=0.3, n_jobs=8, random_state=seed, verbose=-1)
    lg.fit(Xtr, ytr)
    rf = lgb.LGBMRegressor(num_leaves=31, learning_rate=LEARNING_RATE, n_estimators=600, colsample_bytree=0.3,
                           subsample=0.8, subsample_freq=1, min_child_samples=10, n_jobs=8,
                           random_state=seed, verbose=-1)
    rf.fit(Xtr, ytr)
    kr = huber_krr(tanimoto(Xtr[:, :N_BITS], Xtr[:, :N_BITS]), ytr, tanimoto(Xte[:, :N_BITS], Xtr[:, :N_BITS]))
    return np.column_stack([lg.predict(Xte), rf.predict(Xte), kr])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    Ltr = np.array([Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in train_smiles])
    Lte = np.array([Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in test_smiles])
    b, a = np.polyfit(Ltr, y, 1)
    y = y - (a + b * Ltr)
    Xte = featurize(test_smiles)
    Dtr = desc_matrix(train_smiles)
    Dte = desc_matrix(test_smiles)
    lo, hi = np.percentile(Dtr, 1, axis=0), np.percentile(Dtr, 99, axis=0)
    X = np.hstack([X, np.clip(Dtr, lo, hi).astype(np.float32)])
    Xte = np.hstack([Xte, np.clip(Dte, lo, hi).astype(np.float32)])
    oof = np.zeros((len(y), 3))
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
        oof[va] = base_predict(X[tr], y[tr], X[va], seed)
    meta = LinearRegression(positive=True).fit(oof, y)
    return meta.predict(base_predict(X, y, Xte, seed)) + a + b * Lte
