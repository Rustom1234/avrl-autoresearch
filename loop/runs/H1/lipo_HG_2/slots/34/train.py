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


def train_mlp_ensemble(X_train, y_train, X_test, seed):
    X_train_norm = (X_train - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 1e-7)
    X_test_norm = (X_test - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 1e-7)
    
    ensemble_preds = np.zeros((X_test.shape[0],))
    for i in range(5):
        s = seed + i
        # Simple MLP: 2057 -> 256 -> 64 -> 1
        # Using standard weights initialization
        W1 = np.random.randn(X_train.shape[1], 256).astype(np.float32) * np.sqrt(2./X_train.shape[1])
        W2 = np.random.randn(256, 64).astype(np.float32) * np.sqrt(2./256)
        W3 = np.random.randn(64, 1).astype(np.float32) * np.sqrt(2./64)
        
        # Simple SGD/Adam-lite with L1
        lr = 1e-3
        for epoch in range(100):
            batch_idx = np.random.permutation(X_train.shape[0])[:64]
            bx = X_train_norm[batch_idx]
            by = y_train[batch_idx]
            
            # Forward
            z1 = np.dot(bx, W1)
            a1 = np.maximum(0, z1) # ReLU
            z2 = np.dot(a1, W2)
            a2 = np.maximum(0, z2) # ReLU
            y_pred = np.dot(a2, W3)
            
            # Gradients (L1 Loss)
            dy = y_pred - by
            dw3 = np.dot(a2.T, dy)
            da2 = np.dot(dy, W3.T)
            dz2 = da2 * (z2 > 0)
            dw2 = np.dot(a1.T, dz2)
            da1 = np.dot(dz2, W2.T)
            dz1 = da1 * (z1 > 0)
            dw1 = np.dot(bx.T, dz1)
            
            W3 -= lr * (dw3 + 1e-4 * W3)
            W2 -= lr * (dw2 + 1e-4 * W2)
            W1 -= lr * (dw1 + 1e-4 * W1)
            
        z1 = np.dot(X_test_norm, W1)
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, W2)
        a2 = np.maximum(0, z2)
        ensemble_preds += np.dot(a2, W3)
        
    return ensemble_preds / 5


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
        verbose=0
    )
    model.fit(X_train, y_train)
    
    X_test = featurize(test_smiles)
    
    cat_preds = model.predict(X_test)
    mlp_preds = train_mlp_ensemble(X_train, y_train, X_test, seed)
    
    return 0.7 * cat_preds + 0.3 * mlp_preds
