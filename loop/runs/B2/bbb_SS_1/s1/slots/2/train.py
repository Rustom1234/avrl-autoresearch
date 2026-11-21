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


def standardize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    try:
        m = rdMolStandardize.LargestFragmentChooser().choose(mol)
        return rdMolStandardize.Uncharger().uncharge(m)
    except Exception:
        return mol


def featurize(mols):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for m in mols:
        fp = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        rows.append(np.concatenate([desc, fp]))
    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize([standardize(s) for s in train_smiles])
    Xt = featurize([standardize(s) for s in test_smiles])
    y = np.asarray(train_y, dtype=np.int64)
    nd = len(Descriptors.descList)
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
    # Tanimoto kNN (k=10, similarity-weighted votes) on binary Morgan bits
    B = (X[:, nd:] > 0).astype(np.float32)
    Bt = (Xt[:, nd:] > 0).astype(np.float32)
    inter = Bt @ B.T
    sim = inter / (Bt.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-9)
    K = 10
    idx = np.argpartition(-sim, K - 1, axis=1)[:, :K]
    s = np.take_along_axis(sim, idx, 1)
    knn = (s * y[idx]).sum(1) / (s.sum(1) + 1e-9)
    preds.append(rankdata(knn) / len(Xt))
    # MLP on quantile-scaled descriptors, averaged over 3 seeds
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=seed)
    Dtr = qt.fit_transform(X[:, :nd])
    Dte = qt.transform(Xt[:, :nd])
    mlp = np.mean([MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-3, early_stopping=True,
                                 max_iter=500, random_state=seed + k).fit(Dtr, y).predict_proba(Dte)[:, 1]
                   for k in range(3)], axis=0)
    preds.append(rankdata(mlp) / len(Xt))
    return np.mean(preds, axis=0)
