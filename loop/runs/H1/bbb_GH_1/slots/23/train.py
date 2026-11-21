"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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


def balanced_forest_proba(X_train, y, X_test, seed):
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_min = min(len(pos), len(neg))
    proba = np.zeros(X_test.shape[0])
    for _ in range(N_TREES):
        idx = np.concatenate([rng.choice(pos, n_min, replace=False),
                              rng.choice(neg, n_min, replace=False)])
        tree = DecisionTreeClassifier(max_features=MAX_FEATURES, random_state=int(rng.randint(1 << 31)))
        tree.fit(X_train[idx], y[idx])
        proba += tree.predict_proba(X_test)[:, 1]
    return proba / N_TREES


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    rf_proba = balanced_forest_proba(X_train, y, X_test, seed)
    xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, n_jobs=8, random_state=seed)
    xgb.fit(X_train, y)
    return 0.5 * (rf_proba + xgb.predict_proba(X_test)[:, 1])
