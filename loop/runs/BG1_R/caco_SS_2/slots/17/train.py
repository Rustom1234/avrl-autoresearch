"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, AllChem, rdMolDescriptors as rdd
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer

# Settings
N_BITS = 1024
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


_SHAPE = {}


def shape_feats(s, m):
    if s not in _SHAPE:
        out = [np.nan] * 7
        try:
            mh = Chem.AddHs(m)
            p = AllChem.ETKDGv3()
            p.randomSeed = 0
            if AllChem.EmbedMolecule(mh, p) == 0:
                out = [rdd.CalcPMI1(mh), rdd.CalcPMI2(mh), rdd.CalcPMI3(mh), rdd.CalcNPR1(mh),
                       rdd.CalcNPR2(mh), rdd.CalcRadiusOfGyration(mh), rdd.CalcAsphericity(mh)]
        except Exception:
            pass
        _SHAPE[s] = out
    return _SHAPE[s]


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append(np.array([f(m) for _, f in Descriptors.descList] + shape_feats(s, m), dtype=np.float64))
    X = np.array(rows)
    X[~np.isfinite(X)] = 0.0
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.5, min_child_samples=10, verbose=-1, random_state=seed, n_jobs=8)
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    model.fit(Xtr, y)
    p_lgb = model.predict(Xte)
    keep = Xtr.std(axis=0) > 0
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(200, len(Xtr)), random_state=seed)
    Str = qt.fit_transform(Xtr[:, keep])
    Ste = qt.transform(Xte[:, keep])
    svr = SVR(kernel='rbf', C=3, epsilon=0.1, gamma='scale').fit(Str, y)
    return 0.5 * p_lgb + 0.5 * svr.predict(Ste)
