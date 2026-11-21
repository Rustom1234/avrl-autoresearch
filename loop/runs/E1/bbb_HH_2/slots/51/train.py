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
N_RULES = 15
MAX_RULE_LEN = 3
N_CUTS = 10
MIN_COVER = 5


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


def rule_scores(x_tr, y_tr, x_te, seed):
    rng = np.random.RandomState(seed)
    conds = []
    for j in range(x_tr.shape[1]):
        for t in np.unique(np.quantile(x_tr[:, j], np.linspace(0.05, 0.95, N_CUTS))):
            conds.append((j, t, True))
            conds.append((j, t, False))
    order = rng.permutation(len(conds))
    conds = [conds[i] for i in order]
    masks = np.array([(x_tr[:, j] > t) if up else (x_tr[:, j] <= t) for j, t, up in conds])
    tmasks = np.array([(x_te[:, j] > t) if up else (x_te[:, j] <= t) for j, t, up in conds])
    y_bool = y_tr.astype(bool)
    uncovered = np.ones(len(y_tr), dtype=bool)
    unmatched = np.ones(len(x_te), dtype=bool)
    scores = np.full(len(x_te), float(y_tr.mean()))
    for _ in range(N_RULES):
        if uncovered.sum() < MIN_COVER:
            break
        cur_tr = uncovered.copy()
        cur_te = np.ones(len(x_te), dtype=bool)
        best = -1.0
        for _depth in range(MAX_RULE_LEN):
            m = masks & cur_tr
            cover = m.sum(axis=1)
            pos = (m & y_bool).sum(axis=1)
            prec = (pos + 1.0) / (cover + 2.0)
            prec[cover < MIN_COVER] = -1.0
            k = int(np.argmax(prec))
            if prec[k] <= best:
                break
            best = prec[k]
            cur_tr = cur_tr & masks[k]
            cur_te = cur_te & tmasks[k]
        if best < 0:
            break
        hit = unmatched & cur_te
        scores[hit] = best
        unmatched &= ~cur_te
        uncovered &= ~cur_tr
    return scores


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
    rule_pred = rule_scores(featurize(train_smiles), y, featurize(test_smiles), seed)
    total += rankdata(rule_pred)
    total = (total - total.min()) / (total.max() - total.min() + 1e-12)
    return total
