"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import lightgbm as lgb
from rdkit.Chem import Descriptors
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from rdkit.Chem.Scaffolds import MurckoScaffold

# Settings
N_BITS = 2048
RADIUS = 2
W_DESC = 0.6  # weight of the descriptor-based LightGBM rank; fingerprint ExtraTrees gets 1 - W_DESC


_DESC_CACHE = {}


def desc_features(smiles):
    rows = []
    for s in smiles:
        if s not in _DESC_CACHE:
            m = Chem.MolFromSmiles(s)
            _DESC_CACHE[s] = [f(m) for _, f in Descriptors.descList]
        rows.append(_DESC_CACHE[s])
    X = np.array(rows, dtype=np.float64)
    X[~np.isfinite(X)] = 0.0
    return np.clip(X, -1e6, 1e6)


def count_features(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def scaffold_groups(smiles):
    keys = {}
    groups = []
    for s in smiles:
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
        groups.append(keys.setdefault(sc, len(keys)))
    return np.array(groups)


def tanimoto(A, B):
    inter = A @ B.T
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-9)


def base_predict(train_smiles, y, test_smiles, seed):
    # three base learners: RF on Morgan bits, LightGBM on descriptors, Tanimoto kernel ridge
    Xf_tr = (count_features(train_smiles) > 0).astype(np.float64)
    Xf_te = (count_features(test_smiles) > 0).astype(np.float64)

    rf = RandomForestRegressor(n_estimators=300, max_features=0.3, n_jobs=-1, random_state=seed)
    rf.fit(Xf_tr, y)
    p_rf = rf.predict(Xf_te)

    lgbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, min_child_samples=10, random_state=seed, n_jobs=8, verbose=-1)
    lgbm.fit(desc_features(train_smiles), y)
    p_desc = lgbm.predict(desc_features(test_smiles))

    mu = y.mean()
    coef = np.linalg.solve(tanimoto(Xf_tr, Xf_tr) + 1.0 * np.eye(len(y)), y - mu)
    p_krr = tanimoto(Xf_te, Xf_tr) @ coef + mu

    return np.column_stack([p_rf, p_desc, p_krr])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    train_smiles = list(train_smiles)
    test_smiles = list(test_smiles)

    # out-of-fold base predictions on scaffold-grouped inner folds
    groups = scaffold_groups(train_smiles)
    oof = np.zeros((len(y), 3))
    for tr, va in GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups):
        oof[va] = base_predict([train_smiles[i] for i in tr], y[tr], [train_smiles[i] for i in va], seed)

    # non-negative blend weights from the stacked OOF predictions
    w = LinearRegression(positive=True, fit_intercept=False).fit(oof, y).coef_

    P = base_predict(train_smiles, y, test_smiles, seed)
    return P @ w
