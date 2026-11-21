"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Settings
N_BITS = 4096
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.2
MIN_SAMPLES_LEAF = 1


DESC_FUNCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.NumHDonors, Descriptors.NumHAcceptors,
              Descriptors.TPSA, Descriptors.NumRotatableBonds, Descriptors.RingCount]


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fp = np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    desc = np.array([[f(m) for f in DESC_FUNCS] for m in mols], dtype=np.float32)
    desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
    return np.hstack([fp, desc])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    rf = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                n_jobs=-1, random_state=seed)
    rf.fit(X_train, y)
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                        n_jobs=8, random_state=seed, eval_metric="logloss")
    xgb.fit(X_train, y)
    return 0.5 * (rf.predict_proba(X_test)[:, 1] + xgb.predict_proba(X_test)[:, 1])
