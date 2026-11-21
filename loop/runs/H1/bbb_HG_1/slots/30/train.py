"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from catboost import CatBoostClassifier
from rdkit.Chem import Descriptors
from sklearn.naive_bayes import BernoulliNB
from scipy.stats import rankdata

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

def featurize_binary(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    features = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        features.append(fp.astype(np.int64))
    return np.array(features, dtype=np.int64)

def fit_predict(train_smiles, train_y, test_smiles, seed):
    train_feats = featurize(train_smiles)
    test_feats = featurize(test_smiles)
    
    # CatBoost
    model = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=seed,
        verbose=False,
        thread_count=8
    )
    model.fit(train_feats, np.asarray(train_y, dtype=np.int64))
    cb_probs = model.predict_proba(test_feats)[:, 1]
    
    # Bernoulli Naive Bayes
    train_feats_bin = featurize_binary(train_smiles)
    nb_model = BernoulliNB(alpha=1.0, random_state=seed)
    nb_model.fit(train_feats_bin, np.asarray(train_y, dtype=np.int64))
    nb_probs = nb_model.predict_proba(featurize_binary(test_smiles))[:, 1]
    
    # Rank blending
    cb_ranks = rankdata(cb_probs) / len(cb_probs)
    nb_ranks = rankdata(nb_probs) / len(nb_probs)
    
    return 0.8 * cb_ranks + 0.2 * nb_ranks
