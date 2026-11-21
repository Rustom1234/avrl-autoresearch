"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

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


def tanimoto_knn_graph(X, k=10):
    n = X.shape[0]
    inter = X @ X.T
    cnt = X.sum(axis=1)
    sim = inter / np.maximum(cnt[:, None] + cnt[None, :] - inter, 1e-9)
    np.fill_diagonal(sim, 0.0)
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    W = np.zeros((n, n))
    W[rows, idx.ravel()] = sim[rows, idx.ravel()]
    W = np.maximum(W, W.T)
    dinv = 1.0 / np.sqrt(np.maximum(W.sum(axis=1), 1e-9))
    return dinv[:, None] * W * dinv[None, :]


def spread_feature(train_smiles, train_y, test_smiles, seed):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(s) for s in list(train_smiles) + list(test_smiles)]
    X = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float64)
    S = tanimoto_knn_graph(X)
    y = np.asarray(train_y, dtype=np.float64)
    n_tr, n = len(y), X.shape[0]

    def run(labelled):
        Y0 = np.zeros((n, 2))
        Y0[labelled, 1] = y[labelled]
        Y0[labelled, 0] = 1.0 - y[labelled]
        F = Y0.copy()
        for _ in range(30):
            F = 0.8 * (S @ F) + 0.2 * Y0
        return F[:, 1] / (F.sum(axis=1) + 1e-12)

    out = np.zeros(n)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, ho in skf.split(np.zeros(n_tr), y.astype(int)):
        out[ho] = run(tr)[ho]
    out[n_tr:] = run(np.arange(n_tr))[n_tr:]
    return out


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
                           colsample_bytree=0.3, subsample=0.8, subsample_freq=1, min_child_samples=10,
                                  n_jobs=-1, random_state=seed)
    s = spread_feature(train_smiles, train_y, test_smiles, seed)
    n_tr = len(train_smiles)
    X_tr = np.hstack([featurize(train_smiles), s[:n_tr, None]])
    X_te = np.hstack([featurize(test_smiles), s[n_tr:, None]])
    model.fit(X_tr, np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(X_te)[:, 1]
