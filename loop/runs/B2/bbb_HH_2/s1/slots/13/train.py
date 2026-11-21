"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
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


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.int64)
    train_sets = fingerprints(train_smiles)
    test_sets = fingerprints(test_smiles)
    idx_fit, idx_val = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=seed)
    params = dict(max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                  min_child_weight=2, random_state=seed, n_jobs=8)
    total = np.zeros(len(test_smiles))
    for x_tr, x_te in zip(train_sets, test_sets):
        probe = XGBClassifier(n_estimators=2000, eval_metric='auc', early_stopping_rounds=50, **params)
        probe.fit(x_tr[idx_fit], y[idx_fit], eval_set=[(x_tr[idx_val], y[idx_val])])
        n_rounds = max(1, int(round(probe.best_iteration / 0.85)))
        model = XGBClassifier(n_estimators=n_rounds, eval_metric='logloss', **params)
        model.fit(x_tr, y)
        total += rankdata(model.predict_proba(x_te)[:, 1])
    total = (total - total.min()) / (total.max() - total.min() + 1e-12)
    return total
