"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors

# Settings
N_BITS = 2048
RADIUS = 2

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        desc = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), 
                          Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), 
                          Descriptors.NumRotatableBonds(mol), Descriptors.TPSA(mol)])
        feats.append(np.concatenate([fp, desc]))
    return np.array(feats, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    
    model = LGBMRegressor(
        num_leaves=31, 
        learning_rate=0.05, 
        n_estimators=300, 
        random_state=seed, 
        importance_type='gain', 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    X_test = featurize(test_smiles)
    return model.predict(X_test)
