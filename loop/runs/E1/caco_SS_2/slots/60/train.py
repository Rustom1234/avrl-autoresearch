"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, MACCSkeys
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

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


def extra_feats(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=512, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        maccs = np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32)
        fc = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        rows.append(np.concatenate([maccs, fc]))
    return np.array(rows)


def base_preds(Dtr, Ltr, y, Dte, Lte, seed):
    model = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.5, min_child_samples=10, verbose=-1, random_state=seed, n_jobs=8)
    model.fit(Ltr, y)
    p_lgb = model.predict(Lte)
    keep = Dtr.std(axis=0) > 0
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(200, len(Dtr)), random_state=seed)
    Str = qt.fit_transform(Dtr[:, keep])
    Ste = qt.transform(Dte[:, keep])
    p_svr = SVR(kernel='rbf', C=3, epsilon=0.1, gamma='scale').fit(Str, y).predict(Ste)
    et = ExtraTreesRegressor(n_estimators=300, max_features=0.5, random_state=seed, n_jobs=8).fit(Dtr, y)
    return np.column_stack([p_lgb, p_svr, et.predict(Dte)])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Dtr, Dte = featurize(train_smiles), featurize(test_smiles)
    Ltr = np.hstack([Dtr, extra_feats(train_smiles)])
    Lte = np.hstack([Dte, extra_feats(test_smiles)])
    y = np.asarray(train_y, dtype=np.float64)
    oof = np.zeros((len(y), 3))
    for a, b in KFold(5, shuffle=True, random_state=seed).split(Dtr):
        oof[b] = base_preds(Dtr[a], Ltr[a], y[a], Dtr[b], Ltr[b], seed)
    stack = Ridge(alpha=1.0, positive=True).fit(oof, y)
    return stack.predict(base_preds(Dtr, Ltr, y, Dte, Lte, seed))
