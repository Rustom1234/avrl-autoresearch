"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from scipy.stats import rankdata
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.linear_model import ARDRegression, Ridge

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def tanimoto(A, B):
    inter = A @ B.T
    na = A.sum(axis=1)[:, None]
    nb = B.sum(axis=1)[None, :]
    return inter / np.maximum(na + nb - inter, 1e-9)


def descriptors(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([f(m) for _, f in Descriptors.descList])
    D = np.array(rows, dtype=np.float64)
    return np.clip(np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)


def rbf(A, B, gamma):
    sq = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def fit_predict(train_smiles, train_y, test_smiles, seed):
    n = len(train_smiles)
    smi = list(train_smiles) + list(test_smiles)
    X = featurize(smi).astype(np.float64)
    D = descriptors(smi)
    keep = D[:n].std(axis=0) > 1e-9
    D = D[:, keep]
    dmu, dsd = D[:n].mean(axis=0), D[:n].std(axis=0) + 1e-9
    D = np.clip((D - dmu) / dsd, -5.0, 5.0)
    Kt = tanimoto(X, X)
    Kmix = 0.5 * (Kt + rbf(D, D, 1.0 / D.shape[1]))
    y = (rankdata(np.asarray(train_y, dtype=np.float64)) - 0.5) / n

    def members(tr, te):
        yt = y[tr]
        mu, sd = yt.mean(), yt.std() + 1e-9
        z = (yt - mu) / sd
        preds = []
        for K in (Kt, Kmix):
            svr = SVR(kernel="precomputed", C=5.0, epsilon=0.05).fit(K[np.ix_(tr, tr)], z)
            preds.append(svr.predict(K[np.ix_(te, tr)]))
        preds.append(ARDRegression(max_iter=300).fit(D[tr], yt).predict(D[te]))
        lgbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, colsample_bytree=0.3,
                                 subsample=0.8, subsample_freq=1, min_child_samples=5, random_state=seed,
                                 n_jobs=8, verbose=-1)
        preds.append(lgbm.fit(X[tr], yt).predict(X[te]))
        preds.append(Ridge(alpha=10.0).fit(D[tr], yt).predict(D[te]))
        return [rankdata(p) / len(p) for p in preds]

    # Greedy Caruana selection on a seeded 80/20 split of the training fold
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(0.8 * n)
    a, b = perm[:cut], perm[cut:]
    P = np.array(members(a, b))
    yb = y[b]
    counts = np.zeros(len(P))
    total = np.zeros(len(b))
    for _ in range(10):
        scores = [np.corrcoef(rankdata(total + P[j]), rankdata(yb))[0, 1] for j in range(len(P))]
        j = int(np.argmax(scores))
        counts[j] += 1
        total += P[j]
    w = counts / counts.sum()
    Q = np.array(members(np.arange(n), np.arange(n, len(smi))))
    return w @ Q
