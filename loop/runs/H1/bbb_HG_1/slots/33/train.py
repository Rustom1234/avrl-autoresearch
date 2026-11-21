"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from catboost import CatBoostClassifier
from rdkit.Chem import Descriptors
from sklearn.decomposition import TruncatedSVD

# Settings
N_BITS = 2048
RADIUS = 2

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    features = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        # Add basic descriptors: MW, LogP, TPSA
        desc = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol)])
        features.append(np.concatenate([fp, desc]))
    return np.array(features, dtype=np.float32)

def fit_predict(train_smiles, train_y, test_smiles, seed):
    # Standard featurization
    train_feats = featurize(train_smiles)
    test_feats = featurize(test_smiles)
    
    # Extract raw Morgan bit vectors for SVD (first N_BITS columns)
    # Note: featurize returns [fp_bits + descriptors, ...]
    train_morgans = train_feats[:, :N_BITS]
    test_morgans = test_feats[:, :N_BITS]
    
    # Fit SVD on combined to capture shared chemical space
    combined_morgans = np.vstack([train_morgans, test_morgans])
    svd = TruncatedSVD(n_components=32, random_state=seed)
    svd_features = svd.fit_transform(combined_morgans)
    
    # Split SVD features back into train and test sets
    train_svd = svd_features[:len(train_feats)]
    test_svd = svd_features[len(train_feats):]
    
    # Concatenate original features with SVD components
    x_train = np.hstack([train_feats, train_svd])
    x_test = np.hstack([test_feats, test_svd])
    
    model = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=seed,
        verbose=False,
        thread_count=8
    )
    model.fit(x_train, np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(x_test)[:, 1]
