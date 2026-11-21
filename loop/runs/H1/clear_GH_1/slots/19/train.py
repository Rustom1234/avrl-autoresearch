"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


N_DESC = 100


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fp = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    desc = np.array([[fn(m) for _, fn in Descriptors.descList[:N_DESC]] for m in mols], dtype=np.float64)
    desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    return np.hstack([fp, desc.astype(np.float32)])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    use_log = bool(np.all(y > 0))
    yt = np.log1p(y) if use_log else y
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    rf = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                               max_depth=10,
                               n_jobs=-1, random_state=seed)
    rf.fit(X_train, yt)
    xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                       n_jobs=8, random_state=seed)
    xgb.fit(X_train, yt)
    pred = 0.7 * rf.predict(X_test) + 0.3 * xgb.predict(X_test)
    return np.expm1(pred) if use_log else pred
