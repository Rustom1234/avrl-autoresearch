"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    rows = [[f(m) for _, f in Descriptors.descList] for m in mols]
    X = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    return X


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr = featurize(train_smiles)
    Xte = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    et = ExtraTreesClassifier(n_estimators=800, max_features=0.5, min_samples_leaf=1,
                              n_jobs=8, random_state=seed)
    et.fit(Xtr, y)
    p_et = et.predict_proba(Xte)[:, 1]
    cb = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                            random_seed=seed, thread_count=8, verbose=0)
    cb.fit(Xtr, y)
    p_cb = cb.predict_proba(Xte)[:, 1]
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=seed)
    Qtr = qt.fit_transform(Xtr)
    Qte = qt.transform(Xte)
    p_mlp = np.zeros(len(Xte))
    for i in range(5):
        mlp = MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-2, early_stopping=True,
                            random_state=seed + i, max_iter=200)
        mlp.fit(Qtr, y)
        p_mlp += mlp.predict_proba(Qte)[:, 1] / 5
    return (rankdata(p_et) + rankdata(p_cb) + rankdata(p_mlp)) / (3 * len(p_mlp))
