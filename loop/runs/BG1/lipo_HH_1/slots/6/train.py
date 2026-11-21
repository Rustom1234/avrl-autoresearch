"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors, MACCSkeys
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    # Morgan bits followed by 167 MACCS structural keys
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        mk = np.zeros((167,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), mk)
        rows.append(np.concatenate([gen.GetFingerprintAsNumPy(mol).astype(np.float32), mk]))
    return np.array(rows, dtype=np.float32)


_DESC_CACHE = {}


def describe(smiles):
    # RDKit physchem descriptors, cached per SMILES across folds
    if smiles not in _DESC_CACHE:
        mol = Chem.MolFromSmiles(smiles)
        _DESC_CACHE[smiles] = np.array([f(mol) for _, f in Descriptors._descList], dtype=np.float64)
    return _DESC_CACHE[smiles]


def fit_lgb(X, y, X_te, seed):
    # Inner 10% validation split (stratified by target quantile) to pick the tree count, then refit on all data
    bins = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    idx_tr, idx_va = train_test_split(np.arange(len(y)), test_size=0.1, stratify=bins, random_state=seed)
    params = dict(objective='l1', learning_rate=0.05, random_state=seed, n_jobs=8, verbose=-1)
    m = lgb.LGBMRegressor(n_estimators=3000, **params)
    m.fit(X[idx_tr], y[idx_tr], eval_set=[(X[idx_va], y[idx_va])], eval_metric='l1',
          callbacks=[lgb.early_stopping(100, verbose=False)])
    best = m.best_iteration_ if m.best_iteration_ > 0 else 3000
    full = lgb.LGBMRegressor(n_estimators=max(1, int(best * 1.1)), **params)
    full.fit(X, y)
    return full.predict(X_te)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    model = lgb.LGBMRegressor(objective='l1', n_estimators=800, learning_rate=0.03, num_leaves=31,
                              min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                              n_jobs=8, random_state=seed, verbose=-1)
    model.fit(featurize(train_smiles), y)
    p_rf = model.predict(featurize(test_smiles))

    D_tr = np.nan_to_num(np.vstack([describe(s) for s in train_smiles]), nan=0.0, posinf=0.0, neginf=0.0)
    D_te = np.nan_to_num(np.vstack([describe(s) for s in test_smiles]), nan=0.0, posinf=0.0, neginf=0.0)
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=seed)
    Z_tr = qt.fit_transform(D_tr)
    Z_te = qt.transform(D_te)
    svr = SVR(C=1.0, epsilon=0.1, gamma='scale')
    svr.fit(Z_tr, y)
    p_svr = svr.predict(Z_te)
    p_lgb = fit_lgb(featurize(train_smiles), y, featurize(test_smiles), seed)
    return (p_rf + p_svr + p_lgb) / 3.0
