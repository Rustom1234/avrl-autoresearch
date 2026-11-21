"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from rdkit.Chem.Scaffolds import MurckoScaffold

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


def rf_pred(Xa, ya, Xb, seed, n_trees):
    m = RandomForestClassifier(n_estimators=n_trees, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=8, random_state=seed)
    m.fit(Xa, ya)
    return m.predict_proba(Xb)[:, 1]


def mlp_pred(Qa, ya, Qb, seed, n):
    p = np.zeros(len(Qb))
    for i in range(n):
        mlp = MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-2, early_stopping=True,
                            random_state=seed + i, max_iter=200)
        mlp.fit(Qa, ya)
        p += mlp.predict_proba(Qb)[:, 1] / n
    return p


def blend(a, b, w):
    return w * rankdata(a) / len(a) + (1 - w) * rankdata(b) / len(b)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr = featurize(train_smiles)
    Xte = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=seed)
    Qtr = qt.fit_transform(Xtr)
    Qte = qt.transform(Xte)

    scaf = []
    for s in train_smiles:
        try:
            scaf.append(MurckoScaffold.MurckoScaffoldSmiles(smiles=s))
        except Exception:
            scaf.append(s)
    groups = np.unique(scaf, return_inverse=True)[1]
    oof_rf = np.zeros(len(y))
    oof_mlp = np.zeros(len(y))
    for a, b in GroupKFold(n_splits=3).split(Xtr, y, groups):
        oof_rf[b] = rankdata(rf_pred(Xtr[a], y[a], Xtr[b], seed, 100)) / len(b)
        oof_mlp[b] = rankdata(mlp_pred(Qtr[a], y[a], Qtr[b], seed, 2)) / len(b)
    best_w, best_auc = 0.5, -1.0
    for w in (0.2, 0.35, 0.5, 0.65, 0.8):
        auc = roc_auc_score(y, w * oof_rf + (1 - w) * oof_mlp)
        if auc > best_auc:
            best_w, best_auc = w, auc

    p_rf = rf_pred(Xtr, y, Xte, seed, N_TREES)
    p_mlp = mlp_pred(Qtr, y, Qte, seed, 5)
    return blend(p_rf, p_mlp, best_w)
