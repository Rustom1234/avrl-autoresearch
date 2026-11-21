"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Settings
N_BITS = 2048
RADIUS = 2
CAT_ITERATIONS = 500


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    cb = CatBoostClassifier(iterations=CAT_ITERATIONS, verbose=False, random_seed=seed, thread_count=8)
    cb.fit(X_train, y)
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                        n_jobs=8, random_state=seed, eval_metric="logloss")
    xgb.fit(X_train, y)
    return 0.5 * (cb.predict_proba(X_test)[:, 1] + xgb.predict_proba(X_test)[:, 1])
