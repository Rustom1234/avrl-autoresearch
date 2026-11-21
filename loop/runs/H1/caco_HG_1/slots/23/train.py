"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from catboost import CatBoostRegressor
from rdkit import DataStructs
from rdkit.Chem import Descriptors

# Settings
N_BITS = 2048

def featurize(smiles):
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    # Filter out None
    mols = [m for m in mols if m is not None]
    smiles = [s for s, m in zip(smiles, mols) if m is not None]
    
    fps = np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    
    # Basic Physicochemical Descriptors
    desc_list = []
    for m in mols:
        d = [
            Descriptors.MolWt(m),
            Descriptors.MolLogP(m),
            Descriptors.NumHDonors(m),
            Descriptors.NumHAcceptors(m),
            Descriptors.HeavyAtomCount(m),
            Descriptors.TPSA(m)
        ]
        desc_list.append(d)
    
    desc_array = np.array(desc_list, dtype=np.float32)
    # Handle NaNs by replacing with 0
    desc_array = np.nan_to_num(desc_array)
    
    # Concatenate Atom-Pair + Descriptors
    return np.hstack([fps, desc_array])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        loss_function='MAE',
        random_seed=seed,
        verbose=0
    )
    
    model.fit(X_train, np.asarray(train_y, dtype=np.float64))
    return model.predict(X_test)
