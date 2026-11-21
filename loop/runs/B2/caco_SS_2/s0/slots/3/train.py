"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
import lightgbm
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer

# Settings
N_BITS = 1024
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append(np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64))
    X = np.array(rows)
    X[~np.isfinite(X)] = 0.0
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    params = dict(learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.5, min_child_samples=10, verbose=-1, random_state=seed, n_jobs=8)
    idx = np.random.RandomState(seed).permutation(len(y))
    n_val = int(round(0.15 * len(y)))
    va, fi = idx[:n_val], idx[n_val:]
    es = LGBMRegressor(n_estimators=2000, **params)
    es.fit(Xtr[fi], y[fi], eval_set=[(Xtr[va], y[va])], eval_metric='l1',
           callbacks=[lightgbm.early_stopping(100, verbose=False)])
    best = es.best_iteration_ if es.best_iteration_ > 0 else 2000
    model = LGBMRegressor(n_estimators=max(1, int(round(1.1 * best))), **params)
    model.fit(Xtr, y)
    p_lgb = model.predict(Xte)
    keep = Xtr.std(axis=0) > 0
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(200, len(Xtr)), random_state=seed)
    Str = qt.fit_transform(Xtr[:, keep])
    Ste = qt.transform(Xte[:, keep])
    svr = SVR(kernel='rbf', C=3, epsilon=0.1, gamma='scale').fit(Str, y)
    return 0.5 * p_lgb + 0.5 * svr.predict(Ste)
