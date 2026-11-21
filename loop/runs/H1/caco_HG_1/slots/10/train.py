"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import lightgbm as lgb

# Settings
N_BITS = 2048
RADIUS = 2

def featurize(smiles):
    # Morgan fingerprints
    # We use RDKit's Morgan Fingerprint directly for simplicity since we need to combine it
    from rdkit.Chem import rdFingerprintGenerator
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fps = np.array([fp_gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)
    
    # Physchem descriptors
    # TPSA, HBD, LogP
    tpsa = np.array([Descriptors.TPSA(Chem.MolFromSmiles(s)) for s in smiles])
    hbd = np.array([Descriptors.NumHDonors(Chem.MolFromSmiles(s)) for s in smiles])
    logp = np.array([Descriptors.MolLogP(Chem.MolFromSmiles(s)) for s in smiles])
    
    # Combine: [Morgan bits, TPSA, HBD, LogP]
    # We'll put descriptors at the end to make monotone constraints easier to index
    # TPSA index: -3, HBD index: -2, LogP index: -1
    return np.hstack([fps, tpsa.reshape(-1, 1), hbd.reshape(-1, 1), logp.reshape(-1, 1)])

def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    
    # Monotone constraints: 
    # TPSA: non-increasing (-1)
    # HBD: non-increasing (-1)
    # LogP: non-decreasing (1)
    # Indices are -3, -2, -1
    mono_constraints = [None] * (N_BITS + 3)
    mono_constraints[-3] = -1
    mono_constraints[-2] = -1
    mono_constraints[-1] = 1
    
    train_data = lgb.Dataset(X_train, label=np.asarray(train_y, dtype=np.float64))
    
    params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'monotone_constraints': mono_constraints
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        seed=seed
    )
    
    return model.predict(X_test)
