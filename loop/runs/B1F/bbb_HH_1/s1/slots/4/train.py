"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors, Descriptors, Crippen
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def descriptors(mol):
    charge = sum(abs(a.GetFormalCharge()) for a in mol.GetAtoms())
    return [Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumRings(mol), mol.GetNumHeavyAtoms(),
            rdMolDescriptors.CalcFractionCSP3(mol), charge]


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        rows.append(np.concatenate([gen.GetFingerprintAsNumPy(mol), np.array(descriptors(mol), dtype=np.float32)]))
    return np.array(rows, dtype=np.float32)


_DESC_CACHE = {}


def full_descriptors(smiles_list):
    out = []
    for s in smiles_list:
        if s not in _DESC_CACHE:
            mol = Chem.MolFromSmiles(s)
            _DESC_CACHE[s] = [f(mol) for _, f in Descriptors.descList]
        out.append(_DESC_CACHE[s])
    arr = np.array(out, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def scaled_rank(p):
    return (rankdata(p) - 1) / max(len(p) - 1, 1)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    D_train = full_descriptors(train_smiles)
    D_test = full_descriptors(test_smiles)
    lo, hi = np.percentile(D_train, [1, 99], axis=0)
    # View F: Morgan bits only. View D: hand descriptors plus clipped full RDKit descriptor block.
    F_train, F_test = X_train[:, :N_BITS], X_test[:, :N_BITS]
    D_train = np.hstack([X_train[:, N_BITS:], np.clip(D_train, lo, hi)]).astype(np.float32)
    D_test = np.hstack([X_test[:, N_BITS:], np.clip(D_test, lo, hi)]).astype(np.float32)
    y = np.asarray(train_y, dtype=np.int64)
    preds = []
    for Xtr, Xte, rf_mf, gb_cs in [(F_train, F_test, MAX_FEATURES, 0.3), (D_train, D_test, 'sqrt', 0.5)]:
        rf = RandomForestClassifier(n_estimators=N_TREES, max_features=rf_mf, min_samples_leaf=MIN_SAMPLES_LEAF,
                                    n_jobs=-1, random_state=seed)
        rf.fit(Xtr, y)
        gb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, colsample_bytree=gb_cs,
                            random_state=seed, n_jobs=8, verbose=-1)
        gb.fit(Xtr, y)
        preds.append(scaled_rank(rf.predict_proba(Xte)[:, 1]))
        preds.append(scaled_rank(gb.predict_proba(Xte)[:, 1]))
    return np.mean(preds, axis=0)
