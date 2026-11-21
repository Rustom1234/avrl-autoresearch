"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import train_test_split

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


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    X_test = featurize(test_smiles)
    params = dict(objective="binary", learning_rate=0.02, num_leaves=31,
                  colsample_bytree=0.3, subsample=0.8, subsample_freq=1, min_child_samples=10,
                  n_jobs=-1, random_state=seed)
    # Choose the number of rounds on a seeded stratified 15% validation split
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.15, stratify=y, random_state=seed)
    probe = LGBMClassifier(n_estimators=2000, **params)
    probe.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[early_stopping(100, verbose=False)])
    n_rounds = max(1, int(round((probe.best_iteration_ or 2000) * 1.1)))
    # Refit on the full training fold with the chosen round count
    model = LGBMClassifier(n_estimators=n_rounds, **params)
    model.fit(X, y)
    return model.predict_proba(X_test)[:, 1]
