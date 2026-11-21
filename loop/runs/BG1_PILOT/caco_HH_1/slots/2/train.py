"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from sklearn.neural_network import MLPRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def describe(smiles):
    rows = [[f(Chem.MolFromSmiles(s)) for _, f in Descriptors._descList] for s in smiles]
    d = np.array(rows, dtype=np.float64)
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Dtr = describe(train_smiles)
    Dte = describe(test_smiles)
    mu = Dtr.mean(axis=0)
    sd = Dtr.std(axis=0)
    sd[sd == 0] = 1.0
    Ztr = np.clip((Dtr - mu) / sd, -5, 5)
    Zte = np.clip((Dte - mu) / sd, -5, 5)
    y = np.asarray(train_y, dtype=np.float64)
    preds = []
    for k in range(3):
        model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', alpha=1e-2, solver='adam',
                             learning_rate_init=1e-3, early_stopping=True, validation_fraction=0.15,
                             max_iter=500, random_state=seed + k)
        model.fit(Ztr, y)
        preds.append(model.predict(Zte))
    return np.mean(preds, axis=0)
