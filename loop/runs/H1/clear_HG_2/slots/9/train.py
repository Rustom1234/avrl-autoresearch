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

def featurize(smiles, training_data=None):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    feats = []
    feats = []
    
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        
        # Basic descriptors
        desc_list = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), 
                      Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), 
                      Descriptors.NumRotatableBonds(mol), Descriptors.TPSA(mol)]
        
        feats.append(np.concatenate([fp, np.array(desc_list, dtype=np.float32)]))
    return np.array(feats, dtype=np.float32)

# Re-implementing to allow clipping efficiently
def get_full_features(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    data = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        desc_list = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), 
                      Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), 
                      Descriptors.NumRotatableBonds(mol), Descriptors.TPSA(mol)]
        data.append(np.concatenate([fp, np.array(desc_list, dtype=np.float32)]))
    return np.array(data, dtype=np.float32)

def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = get_full_features(train_smiles)
    # Clip features to 1st-99th percentile
    p1, p99 = np.percentile(X_train, [1, 99], axis=0)
    X_train = np.clip(X_train, p1, p99)
    
    model = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03, 
                           subsample=0.8, colsample_bytree=0.5, 
                           min_child_weight=3, random_state=seed, 
                           n_jobs=-1, tree_method='hist')
    model.fit(X_train, np.asarray(train_y, dtype=np.float64))
    
    X_test = get_full_features(test_smiles)
    X_test = np.clip(X_test, p1, p99)
    return model.predict(X_test)
