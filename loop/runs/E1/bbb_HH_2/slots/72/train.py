"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
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


def make_xgb(n_estimators, max_depth, learning_rate, seed):
    return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=0.8,
                         colsample_bytree=0.3, min_child_weight=2, eval_metric='logloss', random_state=seed, n_jobs=8)


def curriculum_fit(x_tr, y, seed):
    # Stage 0: cheap out-of-fold log-loss on the training fold only
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    for fit_idx, hold_idx in skf.split(x_tr, y):
        cheap = make_xgb(150, 4, 0.05, seed)
        cheap.fit(x_tr[fit_idx], y[fit_idx])
        oof[hold_idx] = cheap.predict_proba(x_tr[hold_idx])[:, 1]
    p = np.clip(oof, 1e-7, 1 - 1e-7)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    keep = np.argsort(loss)[:int(0.75 * len(y))]
    # Stage 1: depth-5 model on the easiest 75%
    stage1 = make_xgb(500, 5, 0.05, seed)
    stage1.fit(x_tr[keep], y[keep])
    # Stage 2: continue boosting on the full training set
    stage2 = make_xgb(200, 5, 0.02, seed)
    stage2.fit(x_tr, y, xgb_model=stage1.get_booster())
    return stage2


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.int64)
    train_sets = fingerprints(train_smiles)
    test_sets = fingerprints(test_smiles)
    total = np.zeros(len(test_smiles))
    for x_tr, x_te in zip(train_sets, test_sets):
        model = curriculum_fit(x_tr, y, seed)
        total += rankdata(model.predict_proba(x_te)[:, 1])
    total = (total - total.min()) / (total.max() - total.min() + 1e-12)
    return total
