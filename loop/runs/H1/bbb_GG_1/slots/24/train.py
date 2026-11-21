"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import lightgbm as lgb

# Settings
N_BITS = 2048
RADIUS = 4
N_TREES = 1000
LEARNING_RATE = 0.05
NUM_LEAVES = 31


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    data = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        c = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
        n = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
        o = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
        s_ = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'S')
        data.append(np.concatenate([fp, [c, n, o, s_]]))
    return np.array(data, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = lgb.LGBMClassifier(n_estimators=N_TREES, learning_rate=LEARNING_RATE, num_leaves=NUM_LEAVES,
                                n_jobs=-1, random_state=seed)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(featurize(test_smiles))[:, 1]
