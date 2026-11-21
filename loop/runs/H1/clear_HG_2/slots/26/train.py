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

def featurize(smiles, radius=RADIUS, bits=N_BITS, use_maccs=False):
    if use_maccs:
        # Use Morgan with radius 0 to simulate MACCS-like bit mapping
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=0, fpSize=166)
    else:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    
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
    y_train = np.asarray(train_y, dtype=np.float64)
    
    params = {
        'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05, 
        'subsample': 0.8, 'colsample_bytree': 0.8, 
        'min_child_weight': 3, 'random_state': seed, 
        'n_jobs': -1, 'tree_method': 'hist'
    }
    
    # View 1: Morgan r2, 2048 bits
    X1_train = featurize(train_smiles, radius=2, bits=2048)
    model1 = XGBRegressor(**params)
    model1.fit(X1_train, y_train)
    
    # View 2: Morgan r3, 1024 bits
    X2_train = featurize(train_smiles, radius=3, bits=1024)
    model2 = XGBRegressor(**params)
    model2.fit(X2_train, y_train)
    
    # View 3: MACCS keys
    X3_train = featurize(train_smiles, use_maccs=True)
    model3 = XGBRegressor(**params)
    model3.fit(X3_train, y_train)
    
    # Ensemble predictions
    p1 = model1.predict(featurize(test_smiles, radius=2, bits=2048))
    p2 = model2.predict(featurize(test_smiles, radius=3, bits=1024))
    p3 = model3.predict(featurize(test_smiles, use_maccs=True))
    
    return (p1 + p2 + p3) / 3.0
