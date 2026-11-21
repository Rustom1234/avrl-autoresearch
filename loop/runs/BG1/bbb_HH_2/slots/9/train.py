"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

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


_DESC_CACHE = {}


def descriptor_vector(smiles):
    if smiles not in _DESC_CACHE:
        m = Chem.MolFromSmiles(smiles)
        vals = []
        for _, func in Descriptors.descList:
            try:
                vals.append(float(func(m)))
            except Exception:
                vals.append(0.0)
        _DESC_CACHE[smiles] = np.array(vals, dtype=np.float64)
    return _DESC_CACHE[smiles]


def linear_features(train_smiles, test_smiles):
    d_tr = np.nan_to_num(np.array([descriptor_vector(s) for s in train_smiles]), nan=0.0, posinf=0.0, neginf=0.0)
    d_te = np.nan_to_num(np.array([descriptor_vector(s) for s in test_smiles]), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(d_tr, 1, axis=0), np.percentile(d_tr, 99, axis=0)
    d_tr, d_te = np.clip(d_tr, lo, hi), np.clip(d_te, lo, hi)
    mu, sd = d_tr.mean(axis=0), d_tr.std(axis=0)
    sd[sd == 0] = 1.0
    d_tr, d_te = (d_tr - mu) / sd, (d_te - mu) / sd

    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    f_tr = np.log1p(np.array([morgan.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in train_smiles], dtype=np.float64))
    f_te = np.log1p(np.array([morgan.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in test_smiles], dtype=np.float64))
    return np.hstack([f_tr, d_tr]), np.hstack([f_te, d_te])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.int64)
    x_tr, x_te = linear_features(train_smiles, test_smiles)
    model = LogisticRegression(penalty='l2', C=0.1, max_iter=2000, class_weight='balanced')
    model.fit(x_tr, y)
    return model.predict_proba(x_te)[:, 1]
