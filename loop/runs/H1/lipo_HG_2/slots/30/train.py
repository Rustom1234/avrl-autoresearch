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
ITERATIONS_MAX = 4000
LEARNING_RATE = 0.02
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
    
    # Split for early stopping
    indices = np.arange(len(X_train))
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    split = int(0.9 * len(X_train))
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_sub_train = X_train[train_idx]
    y_sub_train = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    
    # Find best iteration
    model_early = CatBoostRegressor(
        iterations=ITERATIONS_MAX,
        learning_rate=LEARNING_RATE,
        depth=DEPTH,
        loss_function='MAE',
        random_seed=seed,
        thread_count=8,
        verbose=0
    )
    model_early.fit(X_sub_train, y_sub_train, eval_set=(X_val, y_val), use_best_model=True)
    best_iter = model_early.get_best_iteration()
    
    # Refit on all data
    final_model = CatBoostRegressor(
        iterations=int(1.1 * best_iter) if best_iter > 0 else 100,
        learning_rate=LEARNING_RATE,
        depth=DEPTH,
        loss_function='MAE',
        random_seed=seed,
        thread_count=8,
        verbose=0
    )
    final_model.fit(X_train, y_train)
    
    X_test = featurize(test_smiles)
    return final_model.predict(X_test)
