"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from catboost import CatBoostRegressor

# Settings
N_BITS = 2048
RADIUS = 2
ITERATIONS = 1500
LEARNING_RATE = 0.05
DEPTH = 6


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        
        # Physchem descriptors
        descs = [
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NumAromaticRings(mol)
        ]
        fps.append(np.concatenate([fp, np.array(descs, dtype=np.float32)]))
    
    return np.array(fps, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    
    model = CatBoostRegressor(
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        depth=DEPTH,
        loss_function='MAE',
        random_seed=seed,
        thread_count=8,
        verbose=0,
        rsm=0.5,
        l2_leaf_reg=10,
        random_strength=2,
        bootstrap_type='Bernoulli',
        subsample=0.7
    )
    model.fit(X_train, y_train)
    
    X_test = featurize(test_smiles)
    return model.predict(X_test)
