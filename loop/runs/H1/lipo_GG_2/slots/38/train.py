"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

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
N_ESTIMATORS = 1000
LEARNING_RATE = 0.05
NUM_LEAVES = 31


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    features = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: 
            features.append(np.zeros(N_BITS + 4, dtype=np.float32))
            continue
        fp = gen.GetFingerprintAsNumPy(mol)
        # Add 4 descriptors: MolWt, LogP (RDKit estimate), NumDonors, NumAcceptors
        descriptors = np.array([ 
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol)
        ], dtype=np.float32)
        features.append(np.concatenate([fp, descriptors]))
    return np.array(features, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    
    # Remove bits that are constant (zero variance) in the training set
    # We only look at the first N_BITS (the fingerprints)
    variances = np.var(X_train[:, :N_BITS], axis=0)
    keep_indices = np.where(variances > 0)[0]
    
    X_train_red = np.concatenate([X_train[:, keep_indices], X_train[:, N_BITS:]], axis=1)
    X_test_red = np.concatenate([X_test[:, keep_indices], X_test[:, N_BITS:]], axis=1)
    
    train_data = lgb.Dataset(X_train_red, label=y_train)
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': LEARNING_RATE,
        'num_leaves': NUM_LEAVES,
        'verbosity': -1,
        'seed': seed
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=N_ESTIMATORS
    )
    
    return model.predict(X_test_red)
