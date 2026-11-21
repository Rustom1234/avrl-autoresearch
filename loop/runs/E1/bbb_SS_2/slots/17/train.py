"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.preprocessing import QuantileTransformer

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def pharm_fp(mol):
    v = np.zeros(N_BITS)
    for b in Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory).GetOnBits():
        v[b % N_BITS] = 1
    return v


def featurize(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    rows = [[f(m) for _, f in Descriptors.descList] for m in mols]
    X = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    return np.hstack([X, np.array([pharm_fp(m) for m in mols])])


def base_preds(Xa, Qa, ya, Xb, Qb, seed):
    rf = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES,
                                min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=8, random_state=seed).fit(Xa, ya)
    lg = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, colsample_bytree=0.3,
                        subsample=0.8, subsample_freq=1, random_state=seed, n_jobs=8, verbose=-1).fit(Xa, ya)
    sv = SVC(C=1.0, gamma="scale").fit(Qa, ya)
    return np.column_stack([rf.predict_proba(Xb)[:, 1], lg.predict_proba(Xb)[:, 1],
                            sv.decision_function(Qb)])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr = featurize(train_smiles)
    Xte = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=seed)
    Qtr = qt.fit_transform(Xtr)
    Qte = qt.transform(Xte)
    oof = np.zeros((len(y), 3))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for a, b in skf.split(Xtr, y):
        oof[b] = base_preds(Xtr[a], Qtr[a], y[a], Xtr[b], Qtr[b], seed)
    meta = LogisticRegression(C=1.0).fit(oof, y)
    return meta.predict_proba(base_preds(Xtr, Qtr, y, Xte, Qte, seed))[:, 1]
