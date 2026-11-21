"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    scaler = StandardScaler().fit(X_train)
    y = np.asarray(train_y, dtype=np.float64)
    model = MLPRegressor(hidden_layer_sizes=(512, 128), activation='relu', alpha=1e-3, early_stopping=True,
                         validation_fraction=0.1, max_iter=300, learning_rate_init=1e-3, random_state=seed)
    model.fit(scaler.transform(X_train), y)
    pred = model.predict(scaler.transform(X_test))
    return np.clip(pred, y.min(), y.max())
