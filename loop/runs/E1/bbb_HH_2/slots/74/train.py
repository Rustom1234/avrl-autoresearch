"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import rankdata
from xgboost import XGBClassifier

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


LS_K = 10
LS_ALPHA = 0.9
LS_STEPS = 50


def label_spreading_scores(x_tr, y, x_te):
    x = np.vstack([x_tr, x_te]).astype(np.float64)
    n_tr = x_tr.shape[0]
    n = x.shape[0]
    inter = x @ x.T
    cnt = x.sum(axis=1)
    union = cnt[:, None] + cnt[None, :] - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    np.fill_diagonal(sim, 0.0)
    idx = np.argsort(-sim, axis=1)[:, :LS_K]
    rows = np.repeat(np.arange(n), LS_K)
    cols = idx.ravel()
    W = np.zeros((n, n))
    W[rows, cols] = sim[rows, cols] ** 3
    W = np.maximum(W, W.T)
    d = W.sum(axis=1)
    d_inv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    S = W * d_inv[:, None] * d_inv[None, :]
    Y0 = np.full(n, 0.5)
    Y0[:n_tr] = y
    F = Y0.copy()
    for _ in range(LS_STEPS):
        F = LS_ALPHA * (S @ F) + (1 - LS_ALPHA) * Y0
        F[:n_tr] = y
    return F[n_tr:]


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
    total += rankdata(label_spreading_scores(train_sets[0], y, test_sets[0]))
    total = (total - total.min()) / (total.max() - total.min() + 1e-12)
    return total
