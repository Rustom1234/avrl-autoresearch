"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fp = gen.GetFingerprintAsNumPy(m)
        desc = [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.NumHDonors(m)]
        rows.append(np.concatenate([fp, desc]))
    return np.array(rows, dtype=np.float32)


def _base_probs(X_fit, y_fit, X_pred, seed):
    rf = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                n_jobs=-1, random_state=seed)
    rf.fit(X_fit, y_fit)
    xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, n_jobs=8, random_state=seed)
    xgb.fit(X_fit, y_fit)
    return np.column_stack([rf.predict_proba(X_pred)[:, 1], xgb.predict_proba(X_pred)[:, 1]])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    # Out-of-fold base-model probabilities on the training set (avoids leakage into the meta-model)
    meta_X = np.zeros((len(y), 2))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(X_train, y):
        meta_X[va_idx] = _base_probs(X_train[tr_idx], y[tr_idx], X_train[va_idx], seed)
    test_X = _base_probs(X_train, y, X_test, seed)
    meta = RandomForestClassifier(n_estimators=200, min_samples_leaf=10, n_jobs=-1, random_state=seed)
    meta.fit(meta_X, y)
    return meta.predict_proba(test_X)[:, 1]
