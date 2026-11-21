"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


_DESC_CACHE = {}


def describe(smiles):
    # RDKit physchem descriptors, cached per SMILES across folds
    if smiles not in _DESC_CACHE:
        mol = Chem.MolFromSmiles(smiles)
        _DESC_CACHE[smiles] = np.array([f(mol) for _, f in Descriptors._descList], dtype=np.float64)
    return _DESC_CACHE[smiles]


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    model.fit(featurize(train_smiles), y)
    p_rf = model.predict(featurize(test_smiles))

    D_tr = np.nan_to_num(np.vstack([describe(s) for s in train_smiles]), nan=0.0, posinf=0.0, neginf=0.0)
    D_te = np.nan_to_num(np.vstack([describe(s) for s in test_smiles]), nan=0.0, posinf=0.0, neginf=0.0)
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=seed)
    Z_tr = qt.fit_transform(D_tr)
    Z_te = qt.transform(D_te)
    svr = SVR(C=1.0, epsilon=0.1, gamma='scale')
    svr.fit(Z_tr, y)
    p_svr = svr.predict(Z_te)
    return 0.5 * (p_rf + p_svr)
