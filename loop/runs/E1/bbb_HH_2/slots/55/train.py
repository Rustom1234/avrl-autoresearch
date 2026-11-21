"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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


# Pool members: (model kind, index into [morgan, pair, maccs, descriptors])
MEMBERS = [('xgb', 0), ('xgb', 1), ('xgb', 2), ('et', 0), ('et', 1), ('lr', 3)]
N_STEPS = 8


def fit_member(kind, x_tr, y, x_te, seed):
    if kind == 'xgb':
        model = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                              min_child_weight=2, eval_metric='logloss', random_state=seed, n_jobs=8)
    elif kind == 'et':
        model = ExtraTreesClassifier(n_estimators=300, max_features=0.3, min_samples_leaf=2, random_state=seed,
                                     n_jobs=8)
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    model.fit(x_tr, y)
    return model.predict_proba(x_te)[:, 1]


def greedy_select(ranks, y, steps):
    chosen = []
    total = np.zeros(len(y))
    for _ in range(steps):
        best = max(range(len(ranks)), key=lambda m: roc_auc_score(y, total + ranks[m]))
        chosen.append(best)
        total = total + ranks[best]
    return chosen


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.int64)
    sets_tr = fingerprints(train_smiles) + [featurize(train_smiles)]
    sets_te = fingerprints(test_smiles) + [featurize(test_smiles)]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    oof_ranks, test_ranks = [], []
    for kind, f in MEMBERS:
        x_tr, x_te = sets_tr[f], sets_te[f]
        oof = np.zeros(len(y))
        for a, b in skf.split(x_tr, y):
            oof[b] = fit_member(kind, x_tr[a], y[a], x_tr[b], seed)
        oof_ranks.append(rankdata(oof))
        test_ranks.append(rankdata(fit_member(kind, x_tr, y, x_te, seed)))
    chosen = greedy_select(oof_ranks, y, N_STEPS)
    total = sum(test_ranks[m] for m in chosen)
    total = (total - total.min()) / (total.max() - total.min() + 1e-12)
    return total
