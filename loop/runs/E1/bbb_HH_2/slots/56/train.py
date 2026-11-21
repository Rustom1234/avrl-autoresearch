"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1
GATE_C_GRID = (0.01, 0.03, 0.1)
GATE_MAX = 400


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


def l1_gate(x, y, seed):
    xs = StandardScaler().fit_transform(x)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best_c, best_auc = GATE_C_GRID[0], -1.0
    for c in GATE_C_GRID:
        aucs = []
        for tr, va in skf.split(xs, y):
            lr = LogisticRegression(penalty='l1', C=c, solver='liblinear', max_iter=1000, random_state=seed)
            lr.fit(xs[tr], y[tr])
            aucs.append(roc_auc_score(y[va], lr.decision_function(xs[va])))
        if np.mean(aucs) > best_auc:
            best_c, best_auc = c, np.mean(aucs)
    lr = LogisticRegression(penalty='l1', C=best_c, solver='liblinear', max_iter=1000, random_state=seed)
    lr.fit(xs, y)
    coef = np.abs(lr.coef_.ravel())
    idx = np.flatnonzero(coef)
    if len(idx) == 0:
        return np.arange(x.shape[1])
    if len(idx) > GATE_MAX:
        idx = idx[np.argsort(-coef[idx])[:GATE_MAX]]
    return np.sort(idx)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.int64)
    train_sets = fingerprints(train_smiles)
    test_sets = fingerprints(test_smiles)
    keep = l1_gate(np.hstack(train_sets), y, seed)
    offsets = np.cumsum([0] + [x.shape[1] for x in train_sets])
    total = np.zeros(len(test_smiles))
    for i, (x_tr, x_te) in enumerate(zip(train_sets, test_sets)):
        cols = keep[(keep >= offsets[i]) & (keep < offsets[i + 1])] - offsets[i]
        if len(cols) == 0:
            cols = np.arange(x_tr.shape[1])
        model = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                              min_child_weight=2, eval_metric='logloss', random_state=seed, n_jobs=8)
        model.fit(x_tr[:, cols], y)
        total += rankdata(model.predict_proba(x_te[:, cols])[:, 1])
    total = (total - total.min()) / (total.max() - total.min() + 1e-12)
    return total
