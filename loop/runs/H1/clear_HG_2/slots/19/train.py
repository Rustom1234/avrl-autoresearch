"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors

# Settings
N_BITS = 2048
RADIUS = 2

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        desc = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), 
                          Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), 
                          Descriptors.NumRotatableBonds(mol), Descriptors.TPSA(mol)])
        feats.append(np.concatenate([fp, desc]))
    return np.array(feats, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03, 
                           subsample=0.8, colsample_bytree=0.5, 
                           min_child_weight=3, random_state=seed, 
                           n_jobs=-1, tree_method='hist')
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    eps = 1e-6
    y_train_log = np.log10(y_train + eps)

    # Feature selection
    tmp_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, 
                               subsample=0.8, colsample_bytree=0.5, 
                               min_child_weight=3, random_state=seed, 
                               n_jobs=-1, tree_method='hist')
    tmp_model.fit(X_train, y_train_log)
    
    # Indices of 6 descriptors are at the end (X_train.shape[1] - 6 to X_train.shape[1] - 1)
    num_descriptors = 6
    desc_indices = list(range(X_train.shape[1] - num_descriptors, X_train.shape[1]))
    
    importances = tmp_model.feature_importances_
    # Get indices of top 512 bits (excluding descriptors which are always kept)
    bit_indices = np.argsort(importances[:-num_descriptors])[-512:]
    selected_indices = sorted(list(set(bit_indices + desc_indices)))
    
    X_train_reduced = X_train[:, selected_indices]
    X_test_reduced = featurize(test_smiles)[:, selected_indices]
    
    # Final model
    model = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03, 
                           subsample=0.8, colsample_bytree=0.5, 
                           min_child_weight=3, random_state=seed, 
                           n_jobs=-1, tree_method='hist')
    model.fit(X_train_reduced, y_train_log)
    return model.predict(X_test_reduced)
