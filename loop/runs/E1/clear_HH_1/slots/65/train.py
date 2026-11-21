"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from scipy.stats import rankdata
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from rdkit.Chem import BRICS
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.linear_model import ARDRegression
from sklearn.model_selection import KFold

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


def fragment_sets(smiles):
    return [set(BRICS.BRICSDecompose(Chem.MolFromSmiles(s))) for s in smiles]


def binary_matrix(frag_sets, X_bits, vocab):
    M = np.zeros((len(frag_sets), len(vocab)))
    for i, fs in enumerate(frag_sets):
        for f in fs:
            if f in vocab:
                M[i, vocab[f]] = 1.0
    return np.hstack([X_bits > 0, M])


def te_encode(B, y, m=5.0):
    prior = y.mean()
    return (B.T @ y + m * prior) / (B.sum(axis=0) + m), prior


def te_summary(B, enc, prior):
    act = B > 0
    n_act = act.sum(axis=1)
    has = n_act > 0
    mean = (B @ enc) / np.maximum(n_act, 1)
    mx = np.where(act, enc, -np.inf).max(axis=1)
    mn = np.where(act, enc, np.inf).min(axis=1)
    return np.column_stack([np.where(has, mean, prior), np.where(has, mx, prior), np.where(has, mn, prior)])


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
    frag_tr = fragment_sets(train_smiles)
    frag_te = fragment_sets(test_smiles)
    vocab = {f: i for i, f in enumerate(sorted(set().union(*frag_tr)))}
    B = binary_matrix(frag_tr, X, vocab)
    Bt = binary_matrix(frag_te, Xt, vocab)
    TE = np.zeros((len(y), 3))
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(B):
        TE[va] = te_summary(B[va], *te_encode(B[tr], y[tr]))
    TEt = te_summary(Bt, *te_encode(B, y))
    ard = ARDRegression(max_iter=300)
    ard.fit(np.hstack([D, TE]), y)
    ard_pred = ard.predict(np.hstack([Dt, TEt]))
    return 0.7 * rankdata(svr_pred) / len(svr_pred) + 0.3 * rankdata(ard_pred) / len(ard_pred)
