"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler

# Settings
N_BITS = 2048
RADIUS = 2

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
    X_all = featurize(train_smiles)
    y_all = np.asarray(train_y, dtype=np.float64)
    
    model = CatBoostRegressor(
        iterations=1000, 
        learning_rate=0.05, 
        depth=6, 
        loss_function='MAE', 
        random_seed=seed, 
        verbose=0
    )
    model.fit(X_all, y_all)
    
    X_test = featurize(test_smiles)
    return model.predict(X_test)
