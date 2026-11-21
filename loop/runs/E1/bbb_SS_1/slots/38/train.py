"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from scipy.stats import rankdata
from scipy.spatial.distance import cdist
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

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
        fp = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        rows.append(np.concatenate([desc, fp]))
    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    Xt = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    models = [
        RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                               n_jobs=-1, random_state=seed),
        LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.3, n_jobs=8, random_state=seed, verbose=-1),
        CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, rsm=0.2, thread_count=8,
                           random_seed=seed, verbose=0),
    ]
    preds = []
    for m in models:
        m.fit(X, y)
        preds.append(rankdata(m.predict_proba(Xt)[:, 1]) / len(Xt))
    nd = len(Descriptors.descList)
    qt = QuantileTransformer(n_quantiles=200, random_state=seed).fit(X[:, :nd])
    Dtr, Dte = qt.transform(X[:, :nd]), qt.transform(Xt[:, :nd])
    Ftr, Fte = X[:, nd:].astype(np.float64), Xt[:, nd:].astype(np.float64)

    def kern(Da, Fa, Db, Fb):
        s = Fa.sum(1)[:, None] + Fb.sum(1)[None, :]
        d = cdist(Fa, Fb, 'cityblock')
        tani = (s - d) / np.maximum(s + d, 1e-9)
        rbf = np.exp(-cdist(Da, Db, 'sqeuclidean') / nd)
        return 0.5 * tani + 0.5 * rbf

    svc = SVC(kernel='precomputed', C=1.0)
    svc.fit(kern(Dtr, Ftr, Dtr, Ftr), y)
    preds.append(rankdata(svc.decision_function(kern(Dte, Fte, Dtr, Ftr))) / len(Xt))
    return np.mean(preds, axis=0)
