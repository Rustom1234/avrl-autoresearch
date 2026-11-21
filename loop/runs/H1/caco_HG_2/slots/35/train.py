"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from xgboost import XGBRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_ESTIMATORS = 700
LEARNING_RATE = 0.03

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, 
        fpSize=N_BITS, 
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    )
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    # Morgan fingerprints
    fps = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    # RDKit descriptors
    desc = np.array([[Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.NumHDonors(m), 
                               Descriptors.NumHAcceptors(m), Descriptors.TPSA(m)] 
                      for m in mols], dtype=np.float32)
    return np.hstack([fps, desc])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    X_test = featurize(test_smiles)
    
    model = XGBRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=6,
        random_state=seed,
        tree_method='hist'
    )
    model.fit(X_train, y_train)
    
    return model.predict(X_test)
