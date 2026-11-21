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
from sklearn.linear_model import ARDRegression

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


def ngram_features(all_smiles):
    canon = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in all_smiles]
    vocab = {}
    docs = []
    for s in canon:
        counts = {}
        for n in (2, 3, 4):
            for i in range(len(s) - n + 1):
                g = s[i:i + n]
                counts[g] = counts.get(g, 0) + 1
        docs.append(counts)
        for g in counts:
            if g not in vocab:
                vocab[g] = len(vocab)
    M = np.zeros((len(canon), len(vocab)), dtype=np.float64)
    for r, counts in enumerate(docs):
        for g, c in counts.items():
            M[r, vocab[g]] = c
    return M


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
    n = len(train_smiles)
    S = ngram_features(list(train_smiles) + list(test_smiles))
    S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-9)
    K_train = (tanimoto(X, X) + rbf(D, D, gamma) + S[:n] @ S[:n].T) / 3.0
    K_test = (tanimoto(Xt, X) + rbf(Dt, D, gamma) + S[n:] @ S[:n].T) / 3.0
    y = np.asarray(train_y, dtype=np.float64)
    y = (rankdata(y) - 0.5) / len(y)
    mu, sd = y.mean(), y.std() + 1e-9
    model = SVR(kernel="precomputed", C=5.0, epsilon=0.05)
    model.fit(K_train, (y - mu) / sd)
    svr_pred = model.predict(K_test)
    ard = ARDRegression(max_iter=300)
    ard.fit(D, y)
    ard_pred = ard.predict(Dt)
    return 0.7 * rankdata(svr_pred) / len(svr_pred) + 0.3 * rankdata(ard_pred) / len(ard_pred)
