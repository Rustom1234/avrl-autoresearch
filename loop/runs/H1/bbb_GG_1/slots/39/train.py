"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from catboost import CatBoostClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 1000
LEARNING_RATE = 0.05
DEPTH = 6


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
    model = CatBoostClassifier(iterations=N_TREES, learning_rate=LEARNING_RATE, depth=DEPTH, 
                                random_seed=seed, verbose=False)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(featurize(test_smiles))[:, 1]
