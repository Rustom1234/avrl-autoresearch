"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from scipy.optimize import nnls
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


def base_preds(Xtr, y, Xte, seed):
    model = XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.4, reg_lambda=5, n_jobs=8, random_state=seed)
    model.fit(Xtr, y)
    et = ExtraTreesRegressor(n_estimators=500, max_features=0.3, min_samples_leaf=2,
                             n_jobs=8, random_state=seed)
    et.fit(Xtr, y)
    nd = len(Descriptors.descList)
    svr = make_pipeline(StandardScaler(),
                        QuantileTransformer(n_quantiles=min(200, len(y)), output_distribution="normal",
                                            random_state=seed),
                        SVR(kernel="rbf", C=3, epsilon=0.1))
    svr.fit(Xtr[:, :nd], y)
    return np.column_stack([model.predict(Xte), et.predict(Xte), svr.predict(Xte[:, :nd])])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    oof = np.zeros((len(y), 3))
    for a, b in KFold(5, shuffle=True, random_state=seed).split(Xtr):
        oof[b] = base_preds(Xtr[a], y[a], Xtr[b], seed)
    w, _ = nnls(oof, y)
    w = w / w.sum() if w.sum() > 0 else np.full(3, 1 / 3)
    return base_preds(Xtr, y, Xte, seed) @ w
