"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def featurize_rf(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        counts = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        desc = [Descriptors.MolLogP(m), Descriptors.MolWt(m), rdMolDescriptors.CalcTPSA(m),
                rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
                rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
                rdMolDescriptors.CalcFractionCSP3(m)]
        rows.append(np.concatenate([counts, np.array(desc, dtype=np.float32)]))
    return np.array(rows, dtype=np.float32)


def tanimoto(a, b):
    ab = a @ b.T
    return ab / (a.sum(1)[:, None] + b.sum(1)[None, :] - ab + 1e-9)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    y = np.asarray(train_y, dtype=np.float64)
    Xrf, Xrf_te = featurize_rf(train_smiles), featurize_rf(test_smiles)
    model.fit(Xrf, y)
    top = np.argsort(model.feature_importances_)[::-1][:200]
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    model.fit(Xrf[:, top], y)
    rf_pred = model.predict(Xrf_te[:, top])
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    svr = SVR(kernel="precomputed", C=5.0, epsilon=0.1)
    svr.fit(tanimoto(Xtr, Xtr), y)
    svr_pred = svr.predict(tanimoto(Xte, Xtr))
    return 0.5 * rf_pred + 0.5 * svr_pred
