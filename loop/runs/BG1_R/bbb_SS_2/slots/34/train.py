"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
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


_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
_CACHE = {}


def _row(s):
    if s not in _CACHE:
        m = Chem.MolFromSmiles(s)
        d = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        d = np.clip(np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
        _CACHE[s] = np.concatenate([d, _GEN.GetFingerprintAsNumPy(m).astype(np.float64)])
    return _CACHE[s]


def featurize(smiles):
    return np.array([_row(s) for s in smiles])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr = featurize(train_smiles)
    Xte = featurize(test_smiles)
    model = ExtraTreesClassifier(n_estimators=500, max_features=0.3,
                                 min_samples_leaf=1, n_jobs=8, random_state=seed)
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
    return 0.5 * rankdata(p_rf) / len(p_rf) + 0.5 * rankdata(p_mlp) / len(p_mlp)
