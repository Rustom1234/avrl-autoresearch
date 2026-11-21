"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from scipy.stats import rankdata
from scipy.spatial.distance import cdist
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


def sp_features(train_smiles, test_smiles):
    def counts(smiles):
        out = []
        for s in smiles:
            m = Chem.MolFromSmiles(s)
            dm = Chem.GetDistanceMatrix(m)
            atoms = list(m.GetAtoms())
            sym = [a.GetSymbol() for a in atoms]
            c = {}
            for i, a in enumerate(atoms):
                k = ("atom", sym[i], a.GetIsAromatic())
                c[k] = c.get(k, 0) + 1
                for j in range(i + 1, len(atoms)):
                    k = ("pair", tuple(sorted((sym[i], sym[j]))), min(int(dm[i, j]), 10))
                    c[k] = c.get(k, 0) + 1
            out.append(c)
        return out
    Ctr, Cte = counts(train_smiles), counts(test_smiles)
    vocab = {}
    for c in Ctr + Cte:
        for k in c:
            vocab.setdefault(k, len(vocab))

    def dense(C):
        M = np.zeros((len(C), len(vocab)))
        for r, c in enumerate(C):
            for k, v in c.items():
                M[r, vocab[k]] = v
        return M / np.maximum(M.sum(axis=1, keepdims=True), 1e-9)

    return dense(Ctr), dense(Cte)


def hik(A, B):
    # histogram intersection on L1-normalized rows: sum(min(a,b)) = 1 - 0.5 * L1(a,b)
    return 1.0 - 0.5 * cdist(A, B, "cityblock")


def rbf(A, B, gamma):
    sq = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def fit_predict(train_smiles, train_y, test_smiles, seed):
    S, St = sp_features(train_smiles, test_smiles)
    D = descriptors(train_smiles)
    Dt = descriptors(test_smiles)
    keep = D.std(axis=0) > 1e-9
    D, Dt = D[:, keep], Dt[:, keep]
    dmu, dsd = D.mean(axis=0), D.std(axis=0) + 1e-9
    D = np.clip((D - dmu) / dsd, -5.0, 5.0)
    Dt = np.clip((Dt - dmu) / dsd, -5.0, 5.0)
    gamma = 1.0 / D.shape[1]
    K_train = 0.5 * (hik(S, S) + rbf(D, D, gamma))
    K_test = 0.5 * (hik(St, S) + rbf(Dt, D, gamma))
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
