"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Settings
N_BITS = 512
RADIUS = 2


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS, includeChirality=False)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fp = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        rows.append(np.concatenate([desc, fp]))
    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def make_models(seed):
    return [
        RandomForestRegressor(n_estimators=60, max_features=0.2, min_samples_leaf=2, n_jobs=8, random_state=seed),
        lgb.LGBMRegressor(objective="huber", alpha=1.0, learning_rate=0.08, n_estimators=300, colsample_bytree=0.3,
                          min_child_samples=10, subsample=0.8, subsample_freq=1,
                          n_jobs=8, random_state=seed, verbose=-1),
        CatBoostRegressor(iterations=200, learning_rate=0.1, depth=6, loss_function="MAE",
                          thread_count=8, random_seed=seed, verbose=0),
        "svr",
    ]


def fit_one(model, X, y, Xt):
    if isinstance(model, str):
        sc = StandardScaler().fit(X)
        f = lambda A: np.clip(sc.transform(A), -5, 5)
        m = SVR(C=3.0, gamma="scale", epsilon=0.1).fit(f(X), y)
        return m.predict(f(Xt))
    model.fit(X, y)
    return model.predict(Xt)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    Xt = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    n_models = 4
    oof = np.zeros((len(y), n_models))
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
        for j, m in enumerate(make_models(seed)):
            oof[va, j] = fit_one(m, X[tr], y[tr], X[va])
    meta = Ridge(alpha=1.0, positive=True).fit(oof, y)
    test_preds = np.column_stack([fit_one(m, X, y, Xt) for m in make_models(seed)])
    return meta.predict(test_preds)
