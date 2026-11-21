"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import rankdata
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def xgb_features(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=512)
    names = ["MolLogP", "TPSA", "MolWt", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "RingCount",
             "NumAromaticRings", "FractionCSP3", "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge",
             "MinAbsPartialCharge"]
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        d = [getattr(Descriptors, n)(m) for n in names]
        rows.append(np.concatenate([d, gen.GetCountFingerprintAsNumPy(m)]))
    X = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def tanimoto(A, B):
    # min/max kernel for count vectors
    num = np.empty((len(A), len(B)), dtype=np.float64)
    for i in range(len(A)):
        num[i] = np.minimum(A[i], B).sum(1) / np.maximum(np.maximum(A[i], B).sum(1), 1e-9)
    return num


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    model = RandomForestRegressor(n_estimators=300, max_features=0.5, n_jobs=8, random_state=seed)
    model.fit(xgb_features(train_smiles), y)
    p_rf = model.predict(xgb_features(test_smiles))
    svr = SVR(kernel="precomputed", C=5.0, epsilon=0.1)
    svr.fit(tanimoto(Xtr, Xtr), (y - y.mean()) / (y.std() + 1e-9))
    p_svr = svr.predict(tanimoto(Xte, Xtr))
    return rankdata(p_rf) / len(p_rf) + rankdata(p_svr) / len(p_svr)
