"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
    model = LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
                           colsample_bytree=0.3, subsample=0.8, subsample_freq=1, min_child_samples=10,
                                  n_jobs=-1, random_state=seed)
    y = np.asarray(train_y, dtype=np.int64)
    Xtr = featurize(train_smiles)
    Xte = featurize(test_smiles)

    # LightGBM on Morgan counts + descriptors
    model.fit(Xtr, y)
    p_lgb = model.predict_proba(Xte)[:, 1]

    # Random forest on Morgan counts only
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed)
    rf.fit(Xtr[:, :N_BITS], y)
    p_rf = rf.predict_proba(Xte[:, :N_BITS])[:, 1]

    # Logistic regression on standardized descriptors only
    scaler = StandardScaler().fit(Xtr[:, N_BITS:])
    de_tr = np.clip(scaler.transform(Xtr[:, N_BITS:]), -10, 10)
    de_te = np.clip(scaler.transform(Xte[:, N_BITS:]), -10, 10)
    lr = LogisticRegression(C=0.1, max_iter=5000)
    lr.fit(de_tr, y)
    p_lr = lr.predict_proba(de_te)[:, 1]

    # Equal-weight rank average (AUROC depends only on ordering)
    return np.mean([rankdata(p) / len(p) for p in (p_rf, p_lgb, p_lr)], axis=0)
