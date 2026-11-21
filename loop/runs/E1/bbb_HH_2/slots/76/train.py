"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
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
    total = np.zeros(len(test_smiles))
    for x_tr, x_te in zip(train_sets, test_sets):
        model = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                              min_child_weight=2, eval_metric='logloss', random_state=seed, n_jobs=8)
        model.fit(x_tr, y)
        total += rankdata(model.predict_proba(x_te)[:, 1])
    # Local member: similarity-weighted logistic regression over Tanimoto neighbourhoods
    x_desc_tr = featurize(train_smiles)
    x_desc_te = featurize(test_smiles)
    mu = x_desc_tr.mean(axis=0)
    sd = x_desc_tr.std(axis=0) + 1e-6
    z_tr = (x_desc_tr - mu) / sd
    z_te = (x_desc_te - mu) / sd
    b_tr = train_sets[0]
    b_te = test_sets[0]
    inter = b_te @ b_tr.T
    sim = inter / (b_te.sum(axis=1)[:, None] + b_tr.sum(axis=1)[None, :] - inter + 1e-12)
    local = np.zeros(len(test_smiles))
    for i in range(len(test_smiles)):
        idx = np.argsort(-sim[i])[:150]
        w = sim[i, idx] ** 2 + 1e-12
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            local[i] = np.average(yy, weights=w)
        else:
            lr = LogisticRegression(C=0.3, max_iter=200, random_state=seed)
            lr.fit(z_tr[idx], yy, sample_weight=w)
            local[i] = lr.predict_proba(z_te[i:i + 1])[0, 1]
    total += rankdata(local)
    total = (total - total.min()) / (total.max() - total.min() + 1e-12)
    return total
