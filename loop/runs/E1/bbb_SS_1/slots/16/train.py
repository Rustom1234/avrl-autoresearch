"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer

ND = len(Descriptors.descList)


def knn_tanimoto(A, ya, B, k=10):
    A = (A > 0).astype(np.float32)
    B = (B > 0).astype(np.float32)
    inter = B @ A.T
    union = B.sum(1)[:, None] + A.sum(1)[None, :] - inter
    sim = inter / np.maximum(union, 1e-9)
    idx = np.argsort(-sim, axis=1)[:, :k]
    w = np.take_along_axis(sim, idx, axis=1) + 1e-6
    return (w * ya[idx]).sum(1) / w.sum(1)

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
    preds.append(rankdata(knn_tanimoto(X[:, ND:], y, Xt[:, ND:])) / len(Xt))
    qt = QuantileTransformer(n_quantiles=min(1000, len(X)), output_distribution="normal", random_state=seed)
    Zs = qt.fit_transform(X[:, :ND])
    Zt = qt.transform(Xt[:, :ND])
    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-2, early_stopping=True,
                        max_iter=300, random_state=seed)
    mlp.fit(Zs, y)
    preds.append(rankdata(mlp.predict_proba(Zt)[:, 1]) / len(Xt))
    return np.mean(preds, axis=0)
