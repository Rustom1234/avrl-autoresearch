"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from sklearn.ensemble import GradientBoostingRegressor

# Settings
N_BITS = 1024
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


DESC_NAMES = ["MolLogP", "TPSA", "MolWt", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
              "FractionCSP3", "RingCount", "NumAromaticRings", "HeavyAtomCount", "LabuteASA",
              "NumHeteroatoms", "MolMR", "NHOHCount", "NOCount", "NumAliphaticRings"]


def featurize(smiles):
    fns = [getattr(Descriptors, n) for n in DESC_NAMES]
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append(np.array([f(m) for f in fns], dtype=np.float64))
    X = np.nan_to_num(np.array(rows), nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                                      min_samples_leaf=3, random_state=seed)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles))
