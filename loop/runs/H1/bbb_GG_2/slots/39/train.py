"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
import lightgbm as lgb

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
LEARNING_RATE = 0.1
NUM_LEAVES = 31

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    
    desc = np.array([
        [Descriptors.MolWt(m),
         Descriptors.MolLogP(m),
         Descriptors.NumHDonors(m),
         Descriptors.NumHAcceptors(m),
         Descriptors.TPSA(m)] for m in mols], dtype=np.float32)
    
    return np.hstack([fps, desc])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.int64)
    
    # Internal validation for early stopping
    train_idx = np.arange(len(X_train))
    np.random.SeedState.set_state(np.random.RandomState(seed).state) # Ensure reproducible split
    shuffle_idx = np.random.permutation(len(X_train))
    
    split = int(len(X_train) * 0.8)
    train_sub_idx = shuffle_idx[:split]
    val_idx = shuffle_idx[split:]
    
    X_train_sub, X_val = X_train[train_sub_idx], X_train[val_idx]
    y_train_sub, y_val = y_train[train_sub_idx], y_train[val_idx]
    
    h_train = lgb.train(
        lgb.train.RegressionParams(learning_rate=LEARNING_RATE, num_leaves=NUM_LEAVES),
        lgb.Dataset(X_train_sub, label=y_train_sub),
        valid_sets=[lgb.Dataset(X_train_sub, label=y_train_sub), lgb.Dataset(X_val, label=y_val)],
        valid_names=['train', 'eval'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )
    
    best_n = h_train.best_iteration
    
    # Retrain on full data
    final_model = lgb.LGBMClassifier(
        n_estimators=max(1, best_n),
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        n_jobs=-1,
        random_state=seed,
        importance_type='gain'
    )
    final_model.fit(X_train, y_train)
    
    return final_model.predict_proba(featurize(test_smiles))[:, 1]
