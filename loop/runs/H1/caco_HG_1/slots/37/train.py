"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from rdkit import DataStructs
from rdkit.Chem import Descriptors
from scipy.stats import norm
from scipy.stats import rankdata

# Settings
N_BITS = 2048

def featurize(smiles):
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mols = [m for m in mols if m is not None]
    smiles = [s for s, m in zip(smiles, mols) if m is not None]
    
    fps = np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    
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
    desc_array = np.nan_to_num(desc_array)
    
    return np.hstack([fps, desc_array])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    train_y = np.asarray(train_y, dtype=np.float64)
    
    # Scale features
    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().transform(X_test)
    
    # Standardise target
    y_mean = np.mean(train_y)
    y_std = np.std(train_y)
    train_y_scaled = (train_y - y_mean) / (y_std if y_std != 0 else 1.0)
    
    model = MLPRegressor(
        hidden_layer_sizes=(256, 64),
        alpha=1e-2,
        early_stopping=True,
        validation_fraction=0.15,
        max_iter=500,
        random_state=seed
    )
    
    model.fit(X_train_scaled, train_y_scaled)
    
    preds_scaled = model.predict(X_test_scaled)
    
    # Invert target scaling
    return preds_scaled * y_std + y_mean
