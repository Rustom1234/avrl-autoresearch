"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import rankdata
from sklearn.model_selection import KFold
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


def aux_descriptors(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([
            Descriptors.MolLogP(m),
            rdMolDescriptors.CalcTPSA(m),
            Descriptors.MolWt(m),
            rdMolDescriptors.CalcNumHBD(m),
            rdMolDescriptors.CalcNumHBA(m),
        ])
    return np.array(rows, dtype=np.float32)


def aux_features(x_tr, t_tr, x_te, seed):
    # Out-of-fold predictions for training rows, full-fit predictions for test rows.
    oof = np.zeros((len(x_tr), t_tr.shape[1]))
    te = np.zeros((len(x_te), t_tr.shape[1]))
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for j in range(t_tr.shape[1]):
        for tr_idx, va_idx in kf.split(x_tr):
            reg = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.3,
                               random_state=seed, n_jobs=8)
            reg.fit(x_tr[tr_idx], t_tr[tr_idx, j])
            oof[va_idx, j] = reg.predict(x_tr[va_idx])
        reg = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.3,
                           random_state=seed, n_jobs=8)
        reg.fit(x_tr, t_tr[:, j])
        te[:, j] = reg.predict(x_te)
    return oof, te


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.int64)
    train_sets = fingerprints(train_smiles)
    test_sets = fingerprints(test_smiles)
    oof, aux_te = aux_features(train_sets[0], aux_descriptors(train_smiles), test_sets[0], seed)
    train_sets[0] = np.hstack([train_sets[0], oof])
    test_sets[0] = np.hstack([test_sets[0], aux_te])
    total = np.zeros(len(test_smiles))
    for x_tr, x_te in zip(train_sets, test_sets):
        model = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                              min_child_weight=2, eval_metric='logloss', random_state=seed, n_jobs=8)
        model.fit(x_tr, y)
        total += rankdata(model.predict_proba(x_te)[:, 1])
    total = (total - total.min()) / (total.max() - total.min() + 1e-12)
    return total
