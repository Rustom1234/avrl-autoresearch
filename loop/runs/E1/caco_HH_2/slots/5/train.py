"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Settings
N_BITS = 2048
RADIUS = 2
ITERS = 800
DEPTH = 6
LR = 0.05


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fp = np.zeros(N_BITS, dtype=np.float32)
        for k, v in AllChem.GetHashedMorganFingerprint(m, RADIUS, nBits=N_BITS).GetNonzeroElements().items():
            fp[k] = v
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float32)
        rows.append(np.concatenate([fp, desc]))
    return np.nan_to_num(np.array(rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr = featurize(train_smiles)
    Xte = featurize(test_smiles)
    scaler = StandardScaler().fit(Xtr[:, N_BITS:])
    ztr = np.clip(scaler.transform(Xtr[:, N_BITS:]), -5, 5)
    zte = np.clip(scaler.transform(Xte[:, N_BITS:]), -5, 5)
    Xtr = np.hstack([Xtr[:, :N_BITS], ztr])
    Xte = np.hstack([Xte[:, :N_BITS], zte])
    model = MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-2, learning_rate_init=1e-3,
                         early_stopping=True, validation_fraction=0.1, max_iter=500, random_state=seed)
    model.fit(Xtr, np.asarray(train_y, dtype=np.float64))
    return model.predict(Xte)
