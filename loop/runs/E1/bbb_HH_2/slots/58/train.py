"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

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


def make_clf(seed):
    return XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                         min_child_weight=2, eval_metric='logloss', random_state=seed, n_jobs=8)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.int64)
    train_sets = fingerprints(train_smiles)
    test_sets = fingerprints(test_smiles)
    folds = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=seed).split(train_sets[0], y))
    # Stage 1: teacher = three-member XGBoost ensemble; soft out-of-fold targets on the training fold
    oof = np.zeros(len(y))
    teacher_te = np.zeros(len(test_smiles))
    for x_tr, x_te in zip(train_sets, test_sets):
        for tr_idx, va_idx in folds:
            m = make_clf(seed)
            m.fit(x_tr[tr_idx], y[tr_idx])
            oof[va_idx] += m.predict_proba(x_tr[va_idx])[:, 1] / len(train_sets)
        model = make_clf(seed)
        model.fit(x_tr, y)
        teacher_te += rankdata(model.predict_proba(x_te)[:, 1])
    # Stage 2: student regressors trained on the soft teacher targets
    student_te = np.zeros(len(test_smiles))
    for x_tr, x_te in zip(train_sets, test_sets):
        student = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.3, min_child_weight=2, objective='reg:squarederror',
                               random_state=seed, n_jobs=8)
        student.fit(x_tr, oof)
        student_te += rankdata(student.predict(x_te))
    total = 0.5 * rankdata(teacher_te) + 0.5 * rankdata(student_te)
    total = (total - total.min()) / (total.max() - total.min() + 1e-12)
    return total
