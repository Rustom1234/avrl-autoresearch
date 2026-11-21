"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles, mean_fp=None):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    features = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: 
            features.append(np.zeros(N_BITS + 4, dtype=np.float32))
            continue
        fp = gen.GetFingerprintAsNumPy(mol)
        if mean_fp is not None:
            fp = fp - mean_fp
        # Add 4 descriptors: MolWt, LogP (RDKit estimate), NumDonors, NumAcceptors
        descriptors = np.array([ 
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol)
        ], dtype=np.float32)
        features.append(np.concatenate([fp, descriptors]))
    return np.array(features, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    mean_fp = X_train[:, :N_BITS].mean(axis=0).reshape(1, -1)
    X_train_centered = featurize(train_smiles, mean_fp=mean_fp)
    
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    model.fit(X_train_centered, np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles, mean_fp=mean_fp))
