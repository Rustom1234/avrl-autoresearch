"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from sklearn.ensemble import RandomForestClassifier

# Settings
N_BITS = 2048
RADIUS = 2


def featurize(smiles, drop_cols=None):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    macc_gen = MACCSkeys.GenMACCSKeys
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            morgan = gen.GetCountFingerprintAsNumPy(mol)
            maccs = macc_gen(mol)
            feats.append(np.concatenate([morgan, np.array(maccs, dtype=np.float32)]))
    feats = np.array(feats, dtype=np.float32)
    
    if drop_cols is not None:
        # Drop columns that are nearly constant in the training set
        # Only check the Morgan part (first N_BITS columns)
        morgan_part = feats[:, :N_BITS]
        means = np.mean(morgan_part, axis=0)
        to_drop = np.where((means < 0.05) | (means > 0.95))[0]
        # We need to drop from the full feat vector (morgan + maccs)
        full_to_drop = np.concatenate([to_drop, [i + N_BITS for i in range(len(to_drop)) if i < 0]])
        # Correction: only drop columns that exist in the Morgan bit range
        mask = np.ones(feats.shape[1], dtype=bool)
        mask[to_drop] = False
        return feats[:, mask]
    
    return feats


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    morgan_part = X_train[:, :N_BITS]
    means = np.mean(morgan_part, axis=0)
    to_drop = np.where((means < 0.05) | (means > 0.95))[0]
    
    mask = np.ones(X_train.shape[1], dtype=bool)
    mask[to_drop] = False
    X_train_filtered = X_train[:, mask]
    
    model = RandomForestClassifier(n_estimators=1000, max_features='sqrt', min_samples_leaf=2, 
                                  class_weight='balanced_subsample', n_jobs=-1, random_state=seed)
    model.fit(X_train_filtered, np.asarray(train_y, dtype=np.int64))
    
    X_test = featurize(test_smiles)
    return model.predict_proba(X_test[:, mask])[:, 1]
