"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fp = gen.GetFingerprintAsNumPy(m)
        topo = [Descriptors.TPSA(m), Descriptors.MolWt(m), Descriptors.MolLogP(m), rdMolDescriptors.CalcNumHBD(m),
                rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcChi1n(m), rdMolDescriptors.CalcKappa2(m),
                Descriptors.BertzCT(m)]
        rows.append(np.concatenate([fp, np.nan_to_num(np.array(topo, dtype=np.float32))]))
    return np.array(rows, dtype=np.float32)


def base_models(seed):
    return [
        RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                              n_jobs=-1, random_state=seed),
        XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                     n_jobs=8, random_state=seed),
        CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05, verbose=0, thread_count=8,
                          random_seed=seed),
    ]


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    Xt = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    # Out-of-fold base predictions form the meta-model's training inputs (avoids leakage).
    oof = np.zeros((len(y), 3))
    for tr, va in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
        for j, m in enumerate(base_models(seed)):
            m.fit(X[tr], y[tr])
            oof[va, j] = m.predict(X[va])
    # Base models refit on all training data supply the test-time inputs.
    test_pred = np.zeros((len(Xt), 3))
    for j, m in enumerate(base_models(seed)):
        m.fit(X, y)
        test_pred[:, j] = m.predict(Xt)
    meta = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=seed)
    meta.fit(oof, y)
    return meta.predict(test_pred)
