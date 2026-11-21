"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

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
    model = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.5, min_child_samples=10, verbose=-1, random_state=seed, n_jobs=8)
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    model.fit(Xtr, y)
    p_lgb = model.predict(Xte)
    keep = Xtr.std(axis=0) > 0
    Xk_tr, Xk_te = Xtr[:, keep], Xte[:, keep]
    lo, hi = np.percentile(Xk_tr, [1, 99], axis=0)
    Xk_tr, Xk_te = np.clip(Xk_tr, lo, hi), np.clip(Xk_te, lo, hi)
    try:
        pt = PowerTransformer(method='yeo-johnson', standardize=True).fit(Xk_tr)
        Str, Ste = pt.transform(Xk_tr), pt.transform(Xk_te)
        if not (np.isfinite(Str).all() and np.isfinite(Ste).all()):
            raise ValueError('non-finite power transform')
    except Exception:
        qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(200, len(Xtr)), random_state=seed)
        Str = qt.fit_transform(Xk_tr)
        Ste = qt.transform(Xk_te)
    svr = SVR(kernel='rbf', C=3, epsilon=0.1, gamma='scale').fit(Str, y)
    return 0.5 * p_lgb + 0.5 * svr.predict(Ste)
