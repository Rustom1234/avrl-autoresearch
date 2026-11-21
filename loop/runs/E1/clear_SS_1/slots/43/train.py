"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import rankdata
from rdkit.Chem import Descriptors
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import rdReducedGraphs
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer
from scipy.spatial.distance import cdist

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
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=1024)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        d = [f(m) for _, f in Descriptors.descList]
        av = np.array(list(pyAvalonTools.GetAvalonCountFP(m, nBits=1024)), dtype=np.float64)
        erg = np.asarray(rdReducedGraphs.GetErGFingerprint(m), dtype=np.float64)
        rows.append(np.concatenate([d, gen.GetCountFingerprintAsNumPy(m), av, erg]))
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
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    model = XGBRegressor(max_depth=4, n_estimators=600, learning_rate=0.03, subsample=0.7, colsample_bytree=0.3,
                         min_child_weight=3, reg_lambda=5, n_jobs=8, random_state=seed)
    Ftr, Fte = xgb_features(train_smiles), xgb_features(test_smiles)
    model.fit(Ftr, y)
    p_rf = model.predict(Fte)
    nd = len(Descriptors.descList)
    qt = QuantileTransformer(n_quantiles=200, output_distribution="uniform")
    Dtr = np.clip(qt.fit_transform(Ftr[:, :nd]), 0, 1)
    Dte = np.clip(qt.transform(Fte[:, :nd]), 0, 1)
    gamma = 1.0 / (nd * 2)
    Ktr = tanimoto(Xtr, Xtr) * np.exp(-gamma * cdist(Dtr, Dtr, "sqeuclidean"))
    Kte = tanimoto(Xte, Xtr) * np.exp(-gamma * cdist(Dte, Dtr, "sqeuclidean"))
    svr = SVR(kernel="precomputed", C=5.0, epsilon=0.1)
    svr.fit(Ktr, (y - y.mean()) / (y.std() + 1e-9))
    p_svr = svr.predict(Kte)
    return rankdata(p_rf) / len(p_rf) + rankdata(p_svr) / len(p_svr)
