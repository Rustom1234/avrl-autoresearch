"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
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


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr = featurize(train_smiles)
    Xte = featurize(test_smiles)
    nd = len(Descriptors.descList)
    model = ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                                 min_samples_leaf=2, n_jobs=8, random_state=seed)
    y = np.asarray(train_y, dtype=np.int64)
    model.fit(Xtr[:, :nd], y)
    p_rf = model.predict_proba(Xte[:, :nd])[:, 1]
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=seed)
    Qtr = qt.fit_transform(Xtr)
    Qte = qt.transform(Xte)
    p_mlp = np.zeros(len(Xte))
    for i in range(5):
        mlp = MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-2, early_stopping=True,
                            random_state=seed + i, max_iter=200)
        mlp.fit(Qtr, y)
        p_mlp += mlp.predict_proba(Qte)[:, 1] / 5
    return 0.5 * rankdata(p_rf) / len(p_rf) + 0.5 * rankdata(p_mlp) / len(p_mlp)
