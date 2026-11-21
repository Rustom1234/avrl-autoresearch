"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# Settings
N_BITS = 2048
RADIUS = 2

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)

def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    train_y = np.asarray(train_y, dtype=np.float64)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1)
    rf.fit(X_train, train_y)
    rf_pred = rf.predict(X_test)

    # LightGBM
    lgb_train = lgb.Dataset(X_train, label=train_y)
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': seed
    }
    
    trainer = lgb.train(params, lgb_train, num_boost_round=100)
    lgb_pred = trainer.predict(X_test)

    # Blend
    return (rf_pred + lgb_pred) / 2.0
