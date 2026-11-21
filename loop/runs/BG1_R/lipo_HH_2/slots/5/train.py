"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
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
    fp = np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)
    return np.hstack([fp, smarts_counts(smiles)])


def tanimoto(A, B):
    dot = A @ B.T
    na = (A * A).sum(1)[:, None]
    nb = (B * B).sum(1)[None, :]
    return dot / np.maximum(na + nb - dot, 1e-9)


def base_predict(Xtr, ytr, Xte, seed):
    lg = lgb.LGBMRegressor(objective="l1", n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
                           n_jobs=8, random_state=seed, verbose=-1)
    lg.fit(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=150, max_features=0.3, n_jobs=8, random_state=seed)
    rf.fit(Xtr, ytr)
    svr = SVR(kernel="precomputed", C=5.0, epsilon=0.1)
    svr.fit(tanimoto(Xtr[:, :N_BITS], Xtr[:, :N_BITS]), ytr)
    return np.column_stack([lg.predict(Xte), rf.predict(Xte), svr.predict(tanimoto(Xte[:, :N_BITS], Xtr[:, :N_BITS]))])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    Xte = featurize(test_smiles)
    oof = np.zeros((len(y), 3))
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
        oof[va] = base_predict(X[tr], y[tr], X[va], seed)
    meta = LinearRegression(positive=True).fit(oof, y)
    return meta.predict(base_predict(X, y, Xte, seed))
