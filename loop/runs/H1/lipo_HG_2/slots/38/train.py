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
    
    # Residualize on MolLogP
    mols_train = [Chem.MolFromSmiles(s) for s in train_smiles]
    mols_train = [m for m in mols_train if m is not None]
    logp_train = np.array([Descriptors.MolLogP(m) for m in mols_train]).reshape(-1, 1)
    
    # Simple linear regression: y = alpha * logp + beta
    # Using least squares to find residuals
    if len(mols_train) > 0:
        A = np.vstack([logp_train, np.ones(len(logp_train)).reshape(-1, 1)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, y_train, rcond=None)
        y_train_fitted = A @ coeffs
        y_train_res = y_train - y_train_fitted
    else:
        y_train_res = y_train

    model = CatBoostRegressor(
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        depth=DEPTH,
        loss_function='MAE',
        random_seed=seed,
        thread_count=8,
        verbose=0
    )
    model.fit(X_train, y_train_res)
    
    X_test = featurize(test_smiles)
    y_test_res = model.predict(X_test)
    
    # Add baseline back
    mols_test = [Chem.MolFromSmiles(s) for s in test_smiles]
    mols_test = [m for m in mols_test if m is not None]
    if len(mols_test) > 0 and len(mols_train) > 0:
        logp_test = np.array([Descriptors.MolLogP(m) for m in mols_test]).reshape(-1, 1)
        y_test_final = y_test_res + (logp_test @ coeffs.flatten())
        return y_test_final.flatten()
    return y_test_res
