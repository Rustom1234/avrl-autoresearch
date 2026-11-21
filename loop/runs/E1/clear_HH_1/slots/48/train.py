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
from scipy.optimize import minimize
from scipy.special import expit

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
    X = featurize(train_smiles).astype(np.float64)
    Xt = featurize(test_smiles).astype(np.float64)
    D = descriptors(train_smiles)
    Dt = descriptors(test_smiles)
    keep = D.std(axis=0) > 1e-9
    D, Dt = D[:, keep], Dt[:, keep]
    dmu, dsd = D.mean(axis=0), D.std(axis=0) + 1e-9
    D = np.clip((D - dmu) / dsd, -5.0, 5.0)
    Dt = np.clip((Dt - dmu) / dsd, -5.0, 5.0)
    gamma = 1.0 / D.shape[1]
    K_train = 0.5 * (tanimoto(X, X) + rbf(D, D, gamma))
    K_test = 0.5 * (tanimoto(Xt, X) + rbf(Dt, D, gamma))
    y = np.asarray(train_y, dtype=np.float64)
    y = (rankdata(y) - 0.5) / len(y)
    mu, sd = y.mean(), y.std() + 1e-9
    model = SVR(kernel="precomputed", C=5.0, epsilon=0.05)
    model.fit(K_train, (y - mu) / sd)
    svr_pred = model.predict(K_test)
    rng = np.random.default_rng(seed)
    n = len(y)
    pi = rng.integers(0, n, size=200000)
    pj = rng.integers(0, n, size=200000)
    keep_pairs = np.abs(y[pi] - y[pj]) > 0.05
    pi, pj = pi[keep_pairs][:40000], pj[keep_pairs][:40000]
    s = np.sign(y[pi] - y[pj])
    dif = D[pi] - D[pj]
    lam = 1e-2

    def pair_obj(w):
        z = s * (dif @ w)
        loss = np.mean(np.logaddexp(0.0, -z)) + 0.5 * lam * (w @ w)
        grad = -(dif * (s * expit(-z))[:, None]).mean(axis=0) + lam * w
        return loss, grad

    res = minimize(pair_obj, np.zeros(D.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    rank_pred = Dt @ res.x
    return 0.7 * rankdata(svr_pred) / len(svr_pred) + 0.3 * rankdata(rank_pred) / len(rank_pred)
