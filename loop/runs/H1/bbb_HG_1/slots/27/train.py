"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from catboost import CatBoostClassifier
from rdkit.Chem import Descriptors
from sklearn.linear_model import LogisticRegression

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
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.int64)
    X_test = featurize(test_smiles)

    # CatBoost model
    cb_model = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=seed,
        verbose=False,
        thread_count=8
    )
    cb_model.fit(X_train, y_train)
    cb_probs = cb_model.predict_proba(X_test)[:, 1]

    # Pairwise RankSVM-style linear model on fingerprint differences
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    
    num_pairs = min(200000, len(pos_idx) * len(neg_idx))
    if num_pairs > 0:
        rng = np.random.RandomState(seed)
        n_to_sample = min(num_pairs // 2, len(pos_idx), len(neg_idx))
        i_idx = rng.choice(len(pos_idx), n_to_sample, replace=False)
        j_idx = rng.choice(len(neg_idx), n_to_sample, replace=False)
        
        X_diff = []
        y_diff = []
        for i, j in zip(i_idx, j_idx):
            diff = X_train[pos_idx[i]] - X_train[neg_idx[j]]
            X_diff.append(diff)
            y_diff.append(1)
            X_diff.append(-diff)
            y_diff.append(0)
        
        X_diff = np.array(X_diff)
        y_diff = np.array(y_diff)
        
        lr = LogisticRegression(C=0.1, max_iter=2000, fit_intercept=False)
        lr.fit(X_diff, y_diff)
        
        lr_logits = X_test @ lr.coef_[0]
        lr_probs = 1 / (1 + np.exp(-lr_logits))
        return (cb_probs + lr_probs) / 2
    
    return cb_probs
