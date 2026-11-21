"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
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
        try:
            m2 = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.LargestFragmentChooser().choose(m))
            Chem.SanitizeMol(m2)
            m = m2
        except Exception:
            pass
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

    nd = len(Descriptors.descList)
    # Tanimoto kNN on binary Morgan bits
    A = (X[:, nd:] > 0).astype(np.float32)
    B = (Xt[:, nd:] > 0).astype(np.float32)
    inter = B @ A.T
    sim = inter / np.maximum(B.sum(1)[:, None] + A.sum(1)[None, :] - inter, 1e-9)
    idx = np.argsort(-sim, axis=1)[:, :10]
    w = np.take_along_axis(sim, idx, axis=1)
    knn = (w * y[idx]).sum(1) / np.maximum(w.sum(1), 1e-9)
    preds.append(rankdata(knn) / len(Xt))

    # MLP on quantile-scaled descriptors
    qt = QuantileTransformer(n_quantiles=min(200, len(X)), output_distribution="normal", random_state=seed)
    Dq = qt.fit_transform(X[:, :nd])
    Dtq = qt.transform(Xt[:, :nd])
    mp = []
    for s in range(3):
        mlp = MLPClassifier(hidden_layer_sizes=(512, 128), alpha=0.1, early_stopping=True, max_iter=300,
                            random_state=seed + s)
        mlp.fit(Dq, y)
        mp.append(mlp.predict_proba(Dtq)[:, 1])
    preds.append(rankdata(np.mean(mp, axis=0)) / len(Xt))
    return np.mean(preds, axis=0)
