"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics.pairwise import rbf_kernel
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

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
        fp = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        rows.append(np.concatenate([desc, fp]))
    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def minmax_kernel(A, B):
    sa, sb = A.sum(1), B.sum(1)
    K = np.empty((len(A), len(B)), dtype=np.float64)
    for i in range(len(A)):
        mn = np.minimum(A[i], B).sum(1)
        K[i] = mn / np.maximum(sa[i] + sb - mn, 1e-9)
    return K


def kernel_svc(X, y, Xt, seed):
    nd = len(Descriptors.descList)
    qt = QuantileTransformer(n_quantiles=min(500, len(X)), random_state=seed)
    D = qt.fit_transform(X[:, :nd])
    Dt = qt.transform(Xt[:, :nd])
    g = 1.0 / nd
    K = 0.5 * minmax_kernel(X[:, nd:], X[:, nd:]) + 0.5 * rbf_kernel(D, D, gamma=g)
    Kt = 0.5 * minmax_kernel(Xt[:, nd:], X[:, nd:]) + 0.5 * rbf_kernel(Dt, D, gamma=g)
    svc = SVC(kernel='precomputed', C=1.0)
    svc.fit(K, y)
    return svc.decision_function(Kt)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    Xt = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    models = [
        RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                               n_jobs=-1, random_state=seed),
        LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.3, n_jobs=8, random_state=seed, verbose=-1),
        CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, rsm=0.2, thread_count=8,
                           random_seed=seed, verbose=0),
    ]
    preds = []
    for m in models:
        m.fit(X, y)
        preds.append(rankdata(m.predict_proba(Xt)[:, 1]) / len(Xt))
    preds.append(rankdata(kernel_svc(X, y, Xt, seed)) / len(Xt))
    return np.mean(preds, axis=0)
