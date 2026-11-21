"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
import lightgbm as lgb
from sklearn.model_selection import train_test_split

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
    X = featurize(train_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    params = dict(learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.3, random_state=seed, n_jobs=8, verbose=-1)
    # Early stopping on the inner validation split finds the number of boosting rounds.
    probe = lgb.LGBMClassifier(n_estimators=2000, **params)
    probe.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    n_best = max(int(probe.best_iteration_ or 0), 50)
    # Refit on all training data with the selected number of rounds.
    model = lgb.LGBMClassifier(n_estimators=n_best, **params)
    model.fit(X, y)
    return model.predict_proba(featurize(test_smiles))[:, 1]
