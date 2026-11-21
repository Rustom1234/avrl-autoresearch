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


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    macc_gen = MACCSkeys.GenMACCSKeys
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            morgan = gen.GetCountFingerprintAsNumPy(mol)
            maccs = macc_gen(mol)
            feats.append(np.concatenate([morgan, np.array(maccs, dtype=np.float32)]))
    return np.array(feats, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    x_train = featurize(train_smiles)
    x_test = featurize(test_smiles)
    y_train = np.asarray(train_y, dtype=np.int64)

    # Random Forest
    model = RandomForestClassifier(n_estimators=1000, max_features='sqrt', min_samples_leaf=2, 
                                  class_weight='balanced_subsample', n_jobs=-1, random_state=seed)
    model.fit(x_train, y_train)
    rf_probs = model.predict_proba(x_test)[:, 1]
    
    # Laplacian-corrected Naive Bayes
    x_train_bin = (x_train > 0).astype(np.float32)
    x_test_bin = (x_test > 0).astype(np.float32)
    
    # a_j: count of positive molecules with bit j set
    # n_j: count of all molecules with bit j set
    a_j = (x_train_bin.T * (y_train == 1)).sum(axis=1)
    n_j = x_train_bin.sum(axis=0)
    
    # Log-odds scores
    scores = np.log((a_j + 1) / (n_j - a_j + 1))
    
    # Sum scores for bits set in test molecules
    nb_scores = (x_test_bin.T * scores).sum(axis=1)
    
    # Rank-normalize scores to [0, 1]
    def rank_norm(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    
    rf_rank = rank_norm(rf_probs)
    nb_rank = rank_norm(nb_scores)
    
    return 0.7 * rf_rank + 0.3 * nb_rank
