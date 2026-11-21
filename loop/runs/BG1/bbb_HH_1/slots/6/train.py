"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fp = np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    desc = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=np.float64)
    desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    return np.hstack([fp, desc.astype(np.float32)])


def make_lgbm(seed):
    return LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
                          colsample_bytree=0.3, subsample=0.8, subsample_freq=1, min_child_samples=10,
                          n_jobs=-1, random_state=seed)


def make_rf(seed):
    return RandomForestClassifier(n_estimators=300, max_features=0.3, n_jobs=-1, random_state=seed)


def base_predict(X_fit, y_fit, X_pred, seed):
    lgbm = make_lgbm(seed).fit(X_fit, y_fit)
    rf = make_rf(seed).fit(X_fit, y_fit)
    return np.column_stack([lgbm.predict_proba(X_pred)[:, 1], rf.predict_proba(X_pred)[:, 1]])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    # Inner out-of-fold base predictions on the training data
    oof = np.zeros((len(y), 2))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        oof[va] = base_predict(X[tr], y[tr], X[va], seed)
    meta = LogisticRegression(C=1.0).fit(oof, y)
    # Base models refit on all training data supply the test-time columns
    test_cols = base_predict(X, y, X_test, seed)
    return meta.predict_proba(test_cols)[:, 1]
