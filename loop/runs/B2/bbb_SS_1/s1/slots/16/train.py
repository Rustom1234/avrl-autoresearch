"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


_LFC = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def standardize(s):
    m = Chem.MolFromSmiles(s)
    try:
        m = _UNCHARGER.uncharge(_LFC.choose(m))
    except Exception:
        pass
    return m


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = standardize(s)
        fp = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        rows.append(np.concatenate([desc, fp]))
    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def knn_scores(B_tr, y, B_te, k=15):
    inter = B_te @ B_tr.T
    union = B_te.sum(1)[:, None] + B_tr.sum(1)[None, :] - inter
    sim = inter / np.maximum(union, 1e-9)
    idx = np.argsort(-sim, axis=1)[:, :k]
    w = np.take_along_axis(sim, idx, 1)
    return (w * y[idx]).sum(1) / np.maximum(w.sum(1), 1e-9)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    Xt = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    n_desc = len(Descriptors.descList)
    B = (X[:, n_desc:] > 0).astype(np.float32)
    Bt = (Xt[:, n_desc:] > 0).astype(np.float32)
    qt = QuantileTransformer(n_quantiles=min(1000, len(X)), random_state=seed)
    Q = qt.fit_transform(X)
    Qt = qt.transform(Xt)
    models = [
        RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                               n_jobs=-1, random_state=seed),
        LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=15, max_depth=4, min_child_samples=30,
                       reg_lambda=5.0, subsample=0.7, subsample_freq=1, colsample_bytree=0.2,
                       n_jobs=8, random_state=seed, verbose=-1),
        CatBoostClassifier(iterations=800, learning_rate=0.03, depth=4, l2_leaf_reg=10, rsm=0.15, thread_count=8,
                           random_seed=seed, verbose=0),
    ]
    preds = []
    for m in models:
        m.fit(X, y)
        preds.append(rankdata(m.predict_proba(Xt)[:, 1]) / len(Xt))
    preds.append(rankdata(knn_scores(B, y, Bt)) / len(Xt))
    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), early_stopping=True, max_iter=500, random_state=seed)
    mlp.fit(Q, y)
    preds.append(rankdata(mlp.predict_proba(Qt)[:, 1]) / len(Xt))
    return np.mean(preds, axis=0)
