"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from xgboost import XGBRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    # Patterns for common functional groups
    patterns = [
        Chem.MolFromSmarts('[c1]1[c][c][c][c][c]1'),  # Benzene
        Chem.MolFromSmarts('[OX2H]'),                # Hydroxyl
        Chem.MolFromSmarts('[CX3](=O)')              # Carbonyl
    ]
    features = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        
        counts = [mol.GetNumSubstructs(p) for p in patterns]
        # Append NumAtoms, TPSA, and fragment counts
        features.append(np.append(fp, [mol.GetNumAtoms(), Descriptors.TPSA(mol), *counts]))
    return np.array(features, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = XGBRegressor(n_estimators=N_TREES, max_depth=6, 
                            learning_rate=0.1, subsample=0.8, 
                            colsample_bytree=MAX_FEATURES, 
                            n_jobs=-1, random_state=seed)
    y_train = np.log1p(np.asarray(train_y, dtype=np.float64))
    model.fit(featurize(train_smiles), y_train)
    preds = model.predict(featurize(test_smiles))
    return np.expm1(preds)
