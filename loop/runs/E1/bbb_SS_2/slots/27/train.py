"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from scipy.stats import rankdata
from scipy.spatial.distance import cdist
from rdkit.Chem import rdMolDescriptors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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


def morgan_counts(smiles):
    X = np.zeros((len(smiles), N_BITS))
    for i, s in enumerate(smiles):
        fp = rdMolDescriptors.GetHashedMorganFingerprint(Chem.MolFromSmiles(s), RADIUS, nBits=N_BITS)
        for k, v in fp.GetNonzeroElements().items():
            X[i, k] = v
    return X


def minmax_kernel(A, B):
    sa = A.sum(1)[:, None]
    sb = B.sum(1)[None, :]
    l1 = cdist(A, B, metric="cityblock")
    return (sa + sb - l1) / np.maximum(sa + sb + l1, 1e-9)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr = featurize(train_smiles)
    Xte = featurize(test_smiles)
    model = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES,
                                   min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=8, random_state=seed)
    y = np.asarray(train_y, dtype=np.int64)
    model.fit(Xtr, y)
    p_rf = model.predict_proba(Xte)[:, 1]
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=seed)
    Qtr = qt.fit_transform(Xtr)
    Qte = qt.transform(Xte)
    p_mlp = np.zeros(len(Xte))
    for i in range(5):
        mlp = MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-2, early_stopping=True,
                            random_state=seed + i, max_iter=200)
        mlp.fit(Qtr, y)
        p_mlp += mlp.predict_proba(Qte)[:, 1] / 5
    Ctr = morgan_counts(train_smiles)
    Cte = morgan_counts(test_smiles)
    svc = SVC(kernel="precomputed", C=1.0)
    svc.fit(minmax_kernel(Ctr, Ctr), y)
    d_svm = svc.decision_function(minmax_kernel(Cte, Ctr))
    n = len(p_rf)
    return (rankdata(p_rf) + rankdata(p_mlp) + rankdata(d_svm)) / (3 * n)
