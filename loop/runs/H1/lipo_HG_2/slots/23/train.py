"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from catboost import CatBoostRegressor

# Settings
N_BITS = 2048
RADIUS = 2
ITERATIONS = 1000
LEARNING_RATE = 0.05
DEPTH = 6

from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        
        # Physchem descriptors
        descs = [
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NumAromaticRings(mol)
        ]
        fps.append(np.concatenate([fp, np.array(descs, dtype=np.float32)]))
    
    return np.array(fps, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    X_test = featurize(test_smiles)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y_train))
    
    # Base models
    m1 = CatBoostRegressor(iterations=ITERATIONS, learning_rate=LEARNING_RATE, depth=DEPTH, 
                            loss_function='MAE', random_seed=seed, thread_count=8, verbose=0)
    m2 = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=seed, n_jobs=-1)
    m3 = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=seed, n_jobs=-1)
    
    for tr_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        
        m1.fit(X_tr, y_tr)
        m2.fit(X_tr, y_tr)
        m3.fit(X_tr, y_tr)
        
        oof_preds[val_idx] = (m1.predict(X_val) + 
                                m2.predict(X_val) + 
                                m3.predict(X_val)) / 3
    
    # Meta-learner
    meta_model = LinearRegression(positive=True)
    meta_model.fit(np.ones((len(y_train), 3)), oof_preds)
    weights = meta_model.coef_
    
    # Final predictions
    m1.fit(X_train, y_train)
    m2.fit(X_train, y_train)
    m3.fit(X_train, y_train)
    
    preds = (m1.predict(X_test) * weights[0] + 
                m2.predict(X_test) * weights[1] + 
                m3.predict(X_test) * weights[2])
    
    return preds
