"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors, Descriptors, Crippen
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression

N_RFF = 1000
RFF_C = 0.1

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
    X_train = np.hstack([X_train, np.clip(D_train, lo, hi).astype(np.float32)])
    X_test = np.hstack([X_test, np.clip(D_test, lo, hi).astype(np.float32)])
    y = np.asarray(train_y, dtype=np.int64)
    rf = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                n_jobs=-1, random_state=seed)
    rf.fit(X_train, y)
    gb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, colsample_bytree=0.3,
                        random_state=seed, n_jobs=8, verbose=-1)
    gb.fit(X_train, y)
    Dc_train = np.clip(D_train, lo, hi)
    Dc_test = np.clip(D_test, lo, hi)
    qt = QuantileTransformer(n_quantiles=min(1000, len(D_train)), output_distribution='normal', random_state=seed)
    Q_train = qt.fit_transform(Dc_train)
    Q_test = qt.transform(Dc_test)
    sc = StandardScaler().fit(Q_train)
    Z_train = sc.transform(Q_train)
    Z_test = sc.transform(Q_test)
    rng = np.random.RandomState(seed)
    d = Z_train.shape[1]
    gamma = 1.0 / d
    W = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(d, N_RFF))
    b = rng.uniform(0.0, 2.0 * np.pi, size=N_RFF)
    R_train = np.sqrt(2.0 / N_RFF) * np.cos(Z_train @ W + b)
    R_test = np.sqrt(2.0 / N_RFF) * np.cos(Z_test @ W + b)
    lr_rff = LogisticRegression(C=RFF_C, solver='lbfgs', max_iter=2000)
    lr_rff.fit(R_train, y)
    p_rff = lr_rff.predict_proba(R_test)[:, 1]
    return (scaled_rank(rf.predict_proba(X_test)[:, 1]) + scaled_rank(gb.predict_proba(X_test)[:, 1])
            + scaled_rank(p_rff)) / 3.0
