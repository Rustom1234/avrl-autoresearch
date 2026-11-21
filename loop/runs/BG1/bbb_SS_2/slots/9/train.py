"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import rankdata
from rdkit.Chem import Descriptors
import lightgbm as lgb

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def descriptors(smiles):
    names = [d[0] for d in Descriptors._descList]
    X = np.array([[f(Chem.MolFromSmiles(s)) for _, f in Descriptors._descList] for s in smiles], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def count_fp(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def minmax(A, B, chunk=25):
    sa, sb = A.sum(1), B.sum(1)
    K = np.empty((len(A), len(B)), dtype=np.float64)
    for i in range(0, len(A), chunk):
        d = np.abs(A[i:i + chunk, None, :] - B[None, :, :]).sum(2)
        tot = sa[i:i + chunk, None] + sb[None, :]
        K[i:i + chunk] = (tot - d) / (tot + d + 1e-9)
    return K


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.int64)
    Ftr, Fte = featurize(train_smiles), featurize(test_smiles)
    Dtr, Dte = descriptors(train_smiles), descriptors(test_smiles)
    preds = []

    rf = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                n_jobs=-1, random_state=seed)
    rf.fit(np.hstack([Dtr, Ftr]), y)
    preds.append(rf.predict_proba(np.hstack([Dte, Fte]))[:, 1])

    svm = SVC(kernel="precomputed", C=5.0, random_state=seed)
    Ctr, Cte = count_fp(train_smiles), count_fp(test_smiles)
    svm.fit(minmax(Ctr, Ctr), y)
    preds.append(svm.decision_function(minmax(Cte, Ctr)))

    gb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.5, random_state=seed, n_jobs=8, verbose=-1)
    gb.fit(Dtr, y)
    preds.append(gb.predict_proba(Dte)[:, 1])

    return np.mean([rankdata(p) / len(p) for p in preds], axis=0)
