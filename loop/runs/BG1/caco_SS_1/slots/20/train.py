"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer

# Settings
N_BITS = 1024
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        cnt = gen.GetCountFingerprintAsNumPy(m).astype(np.float64)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        rows.append(np.concatenate([desc, cnt]))
    X = np.nan_to_num(np.array(rows), nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def tanimoto_kernel(A, B):
    K = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        mn = np.minimum(A[i], B).sum(1)
        mx = np.maximum(A[i], B).sum(1)
        K[i] = mn / np.maximum(mx, 1e-9)
    return K


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    nd = len(Descriptors.descList)
    rf = RandomForestRegressor(n_estimators=500, max_features=0.3, n_jobs=8, random_state=seed)
    rf.fit(Xtr[:, :nd], y)
    p_rf = rf.predict(Xte[:, :nd])
    Ctr, Cte = Xtr[:, nd:].astype(np.float64), Xte[:, nd:].astype(np.float64)
    mu = y.mean()
    K = tanimoto_kernel(Ctr, Ctr)
    coef = np.linalg.solve(K + 0.5 * np.eye(len(y)), y - mu)
    p_krr = tanimoto_kernel(Cte, Ctr) @ coef + mu
    return 0.5 * p_rf + 0.5 * p_krr
