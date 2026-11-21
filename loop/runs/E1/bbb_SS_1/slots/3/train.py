"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    def make_models():
        return [
            RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES,
                                   min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=-1, random_state=seed),
            LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.3, n_jobs=8, random_state=seed, verbose=-1),
            make_pipeline(StandardScaler(), LogisticRegression(C=0.005, max_iter=300)),
        ]

    def logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    oof = np.zeros((len(y), 3))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        for j, m in enumerate(make_models()):
            m.fit(X[tr], y[tr])
            oof[va, j] = m.predict_proba(X[va])[:, 1]
    test_p = np.zeros((len(Xt), 3))
    for j, m in enumerate(make_models()):
        m.fit(X, y)
        test_p[:, j] = m.predict_proba(Xt)[:, 1]
    meta = LogisticRegression(C=1.0)
    meta.fit(logit(oof), y)
    return meta.predict_proba(logit(test_p))[:, 1]
