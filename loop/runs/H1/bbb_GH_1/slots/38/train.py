"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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


def count_featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append(gen.GetCountFingerprintAsNumPy(m))
    return np.log1p(np.array(rows, dtype=np.float32))


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    rf = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                n_jobs=-1, random_state=seed)
    rf.fit(X_train, y)
    xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, n_jobs=8, random_state=seed)
    xgb.fit(X_train, y)
    C_train = count_featurize(train_smiles)
    C_test = count_featurize(test_smiles)
    lr = LogisticRegression(C=0.1, max_iter=5000)
    lr.fit(C_train, y)
    return (rf.predict_proba(X_test)[:, 1] + xgb.predict_proba(X_test)[:, 1]
            + lr.predict_proba(C_test)[:, 1]) / 3
