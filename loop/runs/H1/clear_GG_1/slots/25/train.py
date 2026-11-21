"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from catboost import CatBoostRegressor

# Settings
N_BITS = 2048
RADIUS = 2
ITERATIONS = 1500
LEARNING_RATE = 0.05
DEPTH = 6
RANDOM_STRENGTH = 1.0


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    features = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        # Append NumAtoms and TPSA
        features.append(np.append(fp, [mol.GetNumAtoms(), Descriptors.TPSA(mol)]))
    return np.array(features, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = CatBoostRegressor(
        iterations=ITERATIONS, 
        learning_rate=LEARNING_RATE, 
        depth=DEPTH, 
        random_strength=RANDOM_STRENGTH, 
        verbose=False, 
        seed=seed)
    y_train = np.log1p(np.asarray(train_y, dtype=np.float64))
    model.fit(featurize(train_smiles), y_train)
    preds = model.predict(featurize(test_smiles))
    return np.expm1(preds)
