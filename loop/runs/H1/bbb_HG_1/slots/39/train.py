"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from catboost import CatBoostClassifier
from rdkit.Chem import Descriptors

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

from rdkit import DataStructs

def fit_predict(train_smiles, train_y, test_smiles, seed):
    features = featurize(train_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    
    # Remove near-duplicates with conflicting labels
    keep = np.ones(len(train_smiles), dtype=bool)
    for i in range(len(train_smiles)):
        fp_i = features[i]
        for j in range(i + 1, len(train_smiles)):
            if y[i] != y[j]:
                sim = DataStructs.TanimotoSimilarity(fp_i, features[j])
                if sim >= 0.9:
                    keep[i] = False
                    keep[j] = False
                    break
                    
    model = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=seed,
        verbose=False,
        thread_count=8
    )
    model.fit(features[keep], y[keep])
    return model.predict_proba(featurize(test_smiles))[:, 1]
