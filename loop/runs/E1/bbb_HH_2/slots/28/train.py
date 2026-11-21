"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import rankdata
from xgboost import XGBClassifier
from scipy.linalg import cho_solve, cholesky
from scipy.special import expit

GP_SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


ACID = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
BASIC_AMINE = Chem.MolFromSmarts('[NX3;!$(N-C=[O,S,N]);!$(N-a);!$(N-S(=O)=O)]')


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([
            rdMolDescriptors.CalcTPSA(m),
            Descriptors.MolWt(m),
            Descriptors.MolLogP(m),
            rdMolDescriptors.CalcNumHBD(m),
            rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumRotatableBonds(m),
            rdMolDescriptors.CalcFractionCSP3(m),
            Chem.GetFormalCharge(m),
            len(m.GetSubstructMatches(ACID)),
            len(m.GetSubstructMatches(BASIC_AMINE)),
        ])
    return np.array(rows, dtype=np.float32)


def fingerprints(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    pair = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)
    x_morgan = np.array([morgan.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    x_pair = np.array([pair.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    x_maccs = np.array([np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32) for m in mols])
    return [x_morgan, x_pair, x_maccs]


def count_morgan(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def tanimoto_gram(a, b):
    out = np.empty((len(a), len(b)), dtype=np.float64)
    for i in range(len(a)):
        mn = np.minimum(a[i], b).sum(axis=1)
        mx = np.maximum(a[i], b).sum(axis=1)
        out[i] = mn / np.maximum(mx, 1e-12)
    return out


def gp_fit(K, y, n_iter=20):
    # Laplace approximation with Newton iterations (Rasmussen & Williams, Alg. 3.1)
    n = len(y)
    f = np.zeros(n)
    for _ in range(n_iter):
        pi = expit(f)
        W = pi * (1 - pi) + 1e-10
        sW = np.sqrt(W)
        B = np.eye(n) + sW[:, None] * K * sW[None, :]
        L = cholesky(B, lower=True)
        b = W * f + (y - pi)
        a = b - sW * cho_solve((L, True), sW * (K @ b))
        f = K @ a
    loglik = np.sum(-y * np.logaddexp(0, -f) - (1 - y) * np.logaddexp(0, f))
    log_q = -0.5 * a @ f + loglik - np.sum(np.log(np.diag(L)))
    return f, log_q


def gp_predict(train_smiles, y, test_smiles):
    x_tr = count_morgan(train_smiles)
    x_te = count_morgan(test_smiles)
    k_tr = tanimoto_gram(x_tr, x_tr)
    k_te = tanimoto_gram(x_te, x_tr)
    best = None
    for s in GP_SCALES:
        f, log_q = gp_fit(s * k_tr, y)
        if best is None or log_q > best[0]:
            best = (log_q, s, f)
    _, s, f = best
    return s * (k_te @ (y - expit(f)))


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.int64)
    train_sets = fingerprints(train_smiles)
    test_sets = fingerprints(test_smiles)
    total = np.zeros(len(test_smiles))
    for x_tr, x_te in zip(train_sets, test_sets):
        model = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                              min_child_weight=2, eval_metric='logloss', random_state=seed, n_jobs=8)
        model.fit(x_tr, y)
        total += rankdata(model.predict_proba(x_te)[:, 1])
    gp_score = gp_predict(train_smiles, y.astype(np.float64), test_smiles)
    total += rankdata(gp_score)
    total = (total - total.min()) / (total.max() - total.min() + 1e-12)
    return total
