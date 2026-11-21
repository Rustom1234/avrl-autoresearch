"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
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


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    X_test = featurize(test_smiles)

    rf = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                               n_jobs=-1, random_state=seed)
    rf.fit(X_train, y_train)
    per_tree = np.stack([tree.predict(X_test) for tree in rf.estimators_])
    p_rf = np.median(per_tree, axis=0)

    xgb = XGBRegressor(n_estimators=N_TREES, learning_rate=0.05, max_depth=6, colsample_bynode=MAX_FEATURES,
                       n_jobs=8, random_state=seed)
    xgb.fit(X_train, y_train)
    p_xgb = xgb.predict(X_test)

    cb = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, random_seed=seed, verbose=0, thread_count=8)
    cb.fit(X_train, y_train)
    p_cb = cb.predict(X_test)

    return (p_rf + p_xgb + p_cb) / 3.0
