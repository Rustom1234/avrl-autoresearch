"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer
import lightgbm as lgb

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 500
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


_CACHE = {}


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        if s in _CACHE:
            rows.append(_CACHE[s])
            continue
        m = Chem.MolFromSmiles(s)
        fp = gen.GetFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e5, 1e5).astype(np.float32)
        _CACHE[s] = np.concatenate([fp, desc])
        rows.append(_CACHE[s])
    return np.array(rows, dtype=np.float32)


def crippen(smiles):
    return np.array([Descriptors.MolLogP(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float64)


def tanimoto(A, B):
    a = A @ B.T
    return a / (A.sum(1)[:, None] + B.sum(1)[None, :] - a + 1e-9)


def base_preds(Xtr, ytr, Xte, seed, n_trees):
    et = ExtraTreesRegressor(n_estimators=n_trees, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                             n_jobs=-1, random_state=seed).fit(Xtr, ytr)
    gb = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.3, min_child_samples=10, n_jobs=8, random_state=seed,
                           verbose=-1).fit(Xtr, ytr)
    Ftr, Fte = Xtr[:, :N_BITS], Xte[:, :N_BITS]
    sv = SVR(kernel="precomputed", C=5.0, epsilon=0.1).fit(tanimoto(Ftr, Ftr), ytr)
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=seed).fit(Xtr[:, N_BITS:])
    Dtr, Dte = qt.transform(Xtr[:, N_BITS:]), qt.transform(Xte[:, N_BITS:])
    mlp = np.mean([MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3, early_stopping=True, max_iter=200,
                                random_state=seed + i).fit(Dtr, ytr).predict(Dte) for i in range(3)], axis=0)
    return np.column_stack([et.predict(Xte), gb.predict(Xte), sv.predict(tanimoto(Fte, Ftr)), mlp])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    ctr, cte = crippen(train_smiles), crippen(test_smiles)
    y = np.asarray(train_y, dtype=np.float64) - ctr
    oof = np.zeros((len(y), 4))
    for tr, va in KFold(3, shuffle=True, random_state=seed).split(Xtr):
        oof[va] = base_preds(Xtr[tr], y[tr], Xtr[va], seed, 100)
    meta = LinearRegression(positive=True, fit_intercept=False).fit(oof, y)
    return meta.predict(base_preds(Xtr, y, Xte, seed, N_TREES)) + cte
