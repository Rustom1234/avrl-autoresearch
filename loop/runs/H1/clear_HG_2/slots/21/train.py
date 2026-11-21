"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from catboost import CatBoostRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors

# Settings
# Note: Using descriptors only as per the task

def featurize(smiles):
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        # Using a set of standard 2D descriptors
        desc = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), 
                  Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), 
                  Descriptors.NumRotatableBonds(mol), Descriptors.TPSA(mol)]
        feats.append(desc)
    feats = np.array(feats, dtype=np.float32)
    # Replace NaN and inf with 0
    feats[np.isnan(feats)] = 0
    feats[np.isinf(feats)] = 0
    return feats


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    eps = 1e-6
    y_train_log = np.log10(y_train + eps)
    
    model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        random_seed=seed,
        verbose=0,
        thread_count=8
    )
    model.fit(X_train, y_train_log)
    return model.predict(featurize(test_smiles))
