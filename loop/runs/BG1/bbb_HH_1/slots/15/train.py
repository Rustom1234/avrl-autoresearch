"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from lightgbm import LGBMClassifier

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
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    n_cols = X_train.shape[1]
    probs = []
    for k in range(6):
        rng = np.random.RandomState(seed + k)
        cols = np.sort(rng.choice(n_cols, n_cols // 2, replace=False))
        model = LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
                               colsample_bytree=1.0, subsample=0.8, subsample_freq=1, min_child_samples=10,
                               n_jobs=-1, random_state=seed + k)
        model.fit(X_train[:, cols], y)
        probs.append(model.predict_proba(X_test[:, cols])[:, 1])
    return np.mean(probs, axis=0)
