"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

MAX_FEATURES_GRID = [0.1, 0.2, 0.3]

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


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    # Mini grid search: inner 3-fold stratified CV on the training data only
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best_mf, best_score = MAX_FEATURES, -1.0
    for mf in MAX_FEATURES_GRID:
        scores = []
        for tr, va in skf.split(X_train, y):
            cv_rf = RandomForestClassifier(n_estimators=N_TREES, max_features=mf, min_samples_leaf=MIN_SAMPLES_LEAF,
                                           n_jobs=-1, random_state=seed)
            cv_rf.fit(X_train[tr], y[tr])
            scores.append(roc_auc_score(y[va], cv_rf.predict_proba(X_train[va])[:, 1]))
        if np.mean(scores) > best_score:
            best_mf, best_score = mf, np.mean(scores)

    rf = RandomForestClassifier(n_estimators=N_TREES, max_features=best_mf, min_samples_leaf=MIN_SAMPLES_LEAF,
                                n_jobs=-1, random_state=seed)
    rf.fit(X_train, y)
    xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, n_jobs=8, random_state=seed)
    xgb.fit(X_train, y)
    return 0.5 * (rf.predict_proba(X_test)[:, 1] + xgb.predict_proba(X_test)[:, 1])
