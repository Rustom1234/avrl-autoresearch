"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors

# Settings
N_BITS = 2048
RADIUS = 2

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    # Filter out None
    mols = [m for m in mols if m is not None]
    smiles = [s for s, m in zip(smiles, mols) if m is not None]
    
    fps = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    
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
    
    # Concatenate Morgan + Descriptors
    return np.hstack([fps, desc_array])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    
    # RandomForest
    rf = RandomForestRegressor(
        n_estimators=200, 
        max_depth=12, 
        min_samples_leaf=3, 
        max_features='sqrt', 
        random_state=seed, 
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }
    
    # Using LightGBM's native MAE objective if possible or just standard reg
    # Since lgb.reg_mae is objective in some versions, we use regression with MAE metric
    # or manual MAE if required. Here we use standard params.
    
    # Re-adjust for standard LightGBM L1/MAE objective if available
    if hasattr(lgb, 'reg_mae'):
        params['objective'] = 'reg_mae'
    
    model_lgb = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        seed=seed
    )
    
    rf_preds = rf.predict(X_test)
    lgb_preds = model_lgb.predict(X_test)
    
    return (rf_preds + lgb_preds) / 2.0
