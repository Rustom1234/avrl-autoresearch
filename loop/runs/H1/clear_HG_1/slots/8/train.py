"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1

def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.zeros(len(Descriptors.CalcMolDescriptorNames()), dtype=np.float32)
    desc_vals = []
    for name in Descriptors.CalcMolDescriptorNames():
        val = Descriptors.CalcMolDescriptor(name, mol)
        if not np.isfinite(val): val = 0.0
        desc_vals.append(val)
    return np.array(desc_vals, dtype=np.float32)

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fps = np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)
    descs = np.array([get_descriptors(s) for s in smiles])
    return np.hstack([fps, descs])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    y_train = np.asarray(train_y, dtype=np.float64)
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)

    if np.all(y_train >= 0):
        y_train_transformed = np.log1p(y_train)
        model.fit(X_train, y_train_transformed)
        preds = model.predict(X_test)
        return np.expm1(preds)
    else:
        model.fit(X_train, y_train)
        return model.predict(X_test)
