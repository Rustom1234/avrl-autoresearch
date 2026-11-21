"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import rankdata
from rdkit.Chem import Descriptors
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer

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
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    model = XGBRegressor(max_depth=4, n_estimators=600, learning_rate=0.03, subsample=0.7, colsample_bytree=0.3,
                         min_child_weight=3, reg_lambda=5, n_jobs=8, random_state=seed)
    Ftr, Fte = xgb_features(train_smiles), xgb_features(test_smiles)
    model.fit(Ftr, y)
    p_rf = model.predict(Fte)
    nd = len(Descriptors.descList)
    qt = QuantileTransformer(n_quantiles=min(200, len(Ftr)), output_distribution="normal", random_state=seed)
    Dtr, Dte = qt.fit_transform(Ftr[:, :nd]), qt.transform(Fte[:, :nd])
    p_mlp = np.zeros(len(Dte))
    for k in range(5):
        mlp = MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, early_stopping=True, max_iter=300,
                           random_state=seed * 10 + k)
        mlp.fit(Dtr, y)
        p_mlp += mlp.predict(Dte) / 5
    svr = SVR(kernel="precomputed", C=5.0, epsilon=0.1)
    svr.fit(tanimoto(Xtr, Xtr), (y - y.mean()) / (y.std() + 1e-9))
    p_svr = svr.predict(tanimoto(Xte, Xtr))
    return rankdata(p_rf) / len(p_rf) + rankdata(p_svr) / len(p_svr) + rankdata(p_mlp) / len(p_mlp)
