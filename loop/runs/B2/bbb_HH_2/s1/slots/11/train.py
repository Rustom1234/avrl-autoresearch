"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors, rdReducedGraphs
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.Avalon import pyAvalonTools
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


GOBBI_BITS = 2048
AVALON_BITS = 1024
ERG_LEN = 315


def _erg(m):
    try:
        return np.asarray(rdReducedGraphs.GetErGFingerprint(m), dtype=np.float32)
    except Exception:
        return np.zeros(ERG_LEN, dtype=np.float32)


def _gobbi(m):
    row = np.zeros(GOBBI_BITS, dtype=np.float32)
    try:
        fp = Generate.Gen2DFingerprint(m, Gobbi_Pharm2D.factory)
        for b in fp.GetOnBits():
            row[b % GOBBI_BITS] = 1
    except Exception:
        pass
    return row


def _avalon(m):
    row = np.zeros(AVALON_BITS, dtype=np.float32)
    try:
        fp = pyAvalonTools.GetAvalonCountFP(m, nBits=AVALON_BITS)
        for k, v in fp.GetNonzeroElements().items():
            row[k] = v
    except Exception:
        pass
    return row


_FP_CACHE = {}


def fingerprints(smiles):
    missing = [s for s in dict.fromkeys(smiles) if s not in _FP_CACHE]
    if missing:
        sets = _compute_fingerprints(missing)
        for i, s in enumerate(missing):
            _FP_CACHE[s] = [x[i] for x in sets]
    return [np.array([_FP_CACHE[s][k] for s in smiles], dtype=np.float32) for k in range(6)]


def _compute_fingerprints(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    pair = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)
    x_morgan = np.array([morgan.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    x_pair = np.array([pair.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    x_maccs = np.array([np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32) for m in mols])
    x_erg = np.array([_erg(m) for m in mols], dtype=np.float32)
    x_gobbi = np.array([_gobbi(m) for m in mols], dtype=np.float32)
    x_avalon = np.array([_avalon(m) for m in mols], dtype=np.float32)
    return [x_morgan, x_pair, x_maccs, x_erg, x_gobbi, x_avalon]


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
    total = (total - total.min()) / (total.max() - total.min() + 1e-12)
    return total
