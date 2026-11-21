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
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


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
    model = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(featurize(test_smiles))[:, 1]
