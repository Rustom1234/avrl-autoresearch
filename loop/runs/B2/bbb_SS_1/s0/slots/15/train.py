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
RADII = (1, 2, 3)
K_NN = 15
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def standardize(smi):
    m = Chem.MolFromSmiles(smi)
    try:
        s = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.LargestFragmentChooser().choose(m))
        return s if s is not None else m
    except Exception:
        return m


def featurize(smiles):
    mols = [standardize(s) for s in smiles]
    gens = [rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=N_BITS) for r in RADII]
    desc = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=np.float64)
    desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    fps = [np.array([g.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32) for g in gens]
    X = np.hstack([desc.astype(np.float32), np.hstack(fps)])
    B = (fps[RADII.index(2)] > 0).astype(np.float32)  # binary radius-2 bits for Tanimoto kNN
    return X, B


def tanimoto_knn(Btr, ytr, Bte, k=K_NN):
    inter = Bte @ Btr.T
    union = Bte.sum(1)[:, None] + Btr.sum(1)[None, :] - inter
    sim = inter / np.maximum(union, 1e-9)
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    s = np.take_along_axis(sim, idx, axis=1)
    return (s * ytr[idx]).sum(1) / (s.sum(1) + 1e-9)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X, B = featurize(train_smiles)
    Xt, Bt = featurize(test_smiles)
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
    preds.append(rankdata(tanimoto_knn(B, y, Bt)) / len(Xt))
    qt = QuantileTransformer(n_quantiles=min(1000, len(X)), output_distribution="normal", random_state=seed)
    Q = qt.fit_transform(X)
    Qt = qt.transform(Xt)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), early_stopping=True, random_state=seed, max_iter=500)
    mlp.fit(Q, y)
    preds.append(rankdata(mlp.predict_proba(Qt)[:, 1]) / len(Xt))
    return np.mean(preds, axis=0)
