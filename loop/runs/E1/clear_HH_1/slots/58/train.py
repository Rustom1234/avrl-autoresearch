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
WL_ITERS = 3


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


def wl_docs(smiles):
    docs = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        labels = [f"{a.GetAtomicNum()}_{int(a.GetIsAromatic())}_{a.GetFormalCharge()}_{a.GetTotalNumHs()}" for a in m.GetAtoms()]
        nbrs = [[] for _ in labels]
        for b in m.GetBonds():
            i, j, bt = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType())
            nbrs[i].append((bt, j))
            nbrs[j].append((bt, i))
        bag = list(labels)
        for _ in range(WL_ITERS):
            labels = [str(hash("|".join([labels[k]] + sorted(f"{bt}:{labels[n]}" for bt, n in nbrs[k])))) for k in range(len(labels))]
            bag.extend(labels)
        docs.append(bag)
    return docs


def wl_matrix(docs, vocab):
    M = np.zeros((len(docs), len(vocab)))
    for r, bag in enumerate(docs):
        for lab in bag:
            c = vocab.get(lab)
            if c is not None:
                M[r, c] += 1.0
    return M


def cosine(A, B):
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    return An @ Bn.T


def rbf(A, B, gamma):
    sq = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def fit_predict(train_smiles, train_y, test_smiles, seed):
    train_docs = wl_docs(train_smiles)
    vocab = {}
    for bag in train_docs:
        for lab in bag:
            vocab.setdefault(lab, len(vocab))
    X = wl_matrix(train_docs, vocab)
    Xt = wl_matrix(wl_docs(test_smiles), vocab)
    D = descriptors(train_smiles)
    Dt = descriptors(test_smiles)
    keep = D.std(axis=0) > 1e-9
    D, Dt = D[:, keep], Dt[:, keep]
    dmu, dsd = D.mean(axis=0), D.std(axis=0) + 1e-9
    D = np.clip((D - dmu) / dsd, -5.0, 5.0)
    Dt = np.clip((Dt - dmu) / dsd, -5.0, 5.0)
    gamma = 1.0 / D.shape[1]
    K_train = 0.5 * (cosine(X, X) + rbf(D, D, gamma))
    K_test = 0.5 * (cosine(Xt, X) + rbf(Dt, D, gamma))
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
