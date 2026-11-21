"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GroupKFold
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


UNCHARGER = rdMolStandardize.Uncharger()


def standardize(smi):
    m = Chem.MolFromSmiles(smi)
    try:
        frags = Chem.GetMolFrags(m, asMols=True)
        big = max(frags, key=lambda f: f.GetNumHeavyAtoms())
        big = UNCHARGER.uncharge(big)
        Chem.SanitizeMol(big)
        return big
    except Exception:
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


def minmax_tanimoto(A, B):
    mn = np.empty((len(A), len(B)), dtype=np.float64)
    for i in range(len(A)):
        mn[i] = np.minimum(A[i], B).sum(axis=1)
    sa = A.sum(axis=1)[:, None]
    sb = B.sum(axis=1)[None, :]
    return mn / np.maximum(sa + sb - mn, 1e-9)


def tanimoto_binary(A, B):
    Ab = (A > 0).astype(np.float64)
    Bb = (B > 0).astype(np.float64)
    inter = Ab @ Bb.T
    union = Ab.sum(axis=1)[:, None] + Bb.sum(axis=1)[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def knn_scores(S, y_tr, k=5):
    idx = np.argsort(-S, axis=1)[:, :k]
    w = np.take_along_axis(S, idx, axis=1)
    return (w * y_tr[idx]).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9)


def scaffold_groups(smiles):
    ids = {}
    groups = []
    for s in smiles:
        try:
            key = MurckoScaffold.MurckoScaffoldSmiles(mol=standardize(s))
        except Exception:
            key = s
        groups.append(ids.setdefault(key, len(ids)))
    return np.array(groups)


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
    Fc, Fct = X[:, nd:].astype(np.float64), Xt[:, nd:].astype(np.float64)
    # Tanimoto kNN on binary Morgan bits
    S = tanimoto_binary(Fct, Fc)
    preds.append(rankdata(knn_scores(S, y, k=5)) / len(Xt))
    # MLP on quantile-scaled features
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=seed)
    Q, Qt = qt.fit_transform(X), qt.transform(Xt)
    mlp = MLPClassifier((256, 64), early_stopping=True, max_iter=500, random_state=seed)
    preds.append(rankdata(mlp.fit(Q, y).predict_proba(Qt)[:, 1]) / len(Xt))
    # Kernel ridge on 0.5 * MinMax-Tanimoto + 0.5 * RBF(standardized descriptors)
    D, Dt = X[:, :nd].astype(np.float64), Xt[:, :nd].astype(np.float64)
    mu, sd = D.mean(axis=0), D.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.clip((D - mu) / sd, -5, 5)
    Zt = np.clip((Dt - mu) / sd, -5, 5)
    Ktr = 0.5 * minmax_tanimoto(Fc, Fc) + 0.5 * rbf_kernel(Z, Z, gamma=1.0 / nd)
    Kte = 0.5 * minmax_tanimoto(Fct, Fc) + 0.5 * rbf_kernel(Zt, Z, gamma=1.0 / nd)
    ypm = 2.0 * y - 1.0
    groups = scaffold_groups(train_smiles)
    best_a, best_s = 1.0, -np.inf
    for a in (0.1, 1.0):
        scores = []
        for tr, va in GroupKFold(n_splits=3).split(Ktr, y, groups):
            kr = KernelRidge(alpha=a, kernel="precomputed").fit(Ktr[np.ix_(tr, tr)], ypm[tr])
            scores.append(roc_auc_score(y[va], kr.predict(Ktr[np.ix_(va, tr)])))
        if np.mean(scores) > best_s:
            best_a, best_s = a, np.mean(scores)
    kr = KernelRidge(alpha=best_a, kernel="precomputed").fit(Ktr, ypm)
    preds.append(rankdata(kr.predict(Kte)) / len(Xt))
    return np.mean(preds, axis=0)
