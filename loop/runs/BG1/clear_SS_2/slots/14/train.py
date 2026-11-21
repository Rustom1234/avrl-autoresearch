"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from scipy.stats import rankdata
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor

# Settings
N_BITS = 1024
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fp = gen.GetCountFingerprintAsNumPy(m).astype(np.float64)
        desc = np.array([f(m) for _, f in Descriptors._descList], dtype=np.float64)
        rows.append(np.concatenate([desc, fp]))
    X = np.array(rows)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def _minmax_kernel(A, B):
    K = np.empty((len(A), len(B)), dtype=np.float64)
    for i in range(len(A)):
        a = A[i][None, :]
        K[i] = np.minimum(a, B).sum(1) / np.maximum(np.maximum(a, B).sum(1), 1e-9)
    return K


def fit_predict(train_smiles, train_y, test_smiles, seed):
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LinearRegression
    from rdkit.Chem.Scaffolds import MurckoScaffold
    y = np.asarray(train_y, dtype=np.float64)
    n = len(y)
    X = featurize(list(train_smiles) + list(test_smiles))
    nd = len(Descriptors._descList)
    F = X[:, nd:]
    K = _minmax_kernel(F, F[:n])  # (n_all, n)
    groups = [MurckoScaffold.MurckoScaffoldSmiles(s) for s in train_smiles]

    def bases(tr, te):
        rf = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, n_jobs=-1, random_state=seed)
        rf.fit(F[tr], y[tr])
        gbm = LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.5, min_child_samples=10, n_jobs=8, random_state=seed, verbose=-1)
        gbm.fit(X[tr, :nd], y[tr])
        Ktt = K[tr][:, tr]
        mu = y[tr].mean()
        coef = np.linalg.solve(Ktt + 0.5 * np.eye(len(tr)), y[tr] - mu)
        kr = K[te][:, tr] @ coef + mu
        return np.column_stack([rf.predict(F[te]), gbm.predict(X[te, :nd]), kr])

    oof = np.zeros((n, 3))
    for tr, te in GroupKFold(5).split(np.arange(n), groups=groups):
        oof[te] = bases(tr, te)
    R = np.column_stack([rankdata(oof[:, j]) / n for j in range(3)])
    w = LinearRegression(positive=True).fit(R, rankdata(y) / n).coef_
    w = w / w.sum() if w.sum() > 0 else np.ones(3) / 3
    P = bases(np.arange(n), np.arange(n, len(X)))
    return sum(w[j] * rankdata(P[:, j]) for j in range(3))


def _old_fit_predict(train_smiles, train_y, test_smiles, seed):
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    y = np.asarray(train_y, dtype=np.float64)
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    nd = len(Descriptors._descList)
    model.fit(Xtr[:, nd:], y)
    p_rf = model.predict(Xte[:, nd:])
    gbm = LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.5, min_child_samples=10, n_jobs=8, random_state=seed, verbose=-1)
    gbm.fit(Xtr[:, :nd], y)
    p_gbm = gbm.predict(Xte[:, :nd])
    return 0.5 * rankdata(p_rf) + 0.5 * rankdata(p_gbm)
