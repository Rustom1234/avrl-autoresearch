"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors

# Settings
N_BITS = 2048
RADIUS = 2

def featurize(smiles):
    gen_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    gen_atom_pair = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024)
    gen_torsion = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024)
    
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp_morgan = gen_morgan.GetFingerprintAsNumPy(mol)
        fp_atom_pair = gen_atom_pair.GetFingerprintAsNumPy(mol)
        fp_torsion = gen_torsion.GetFingerprintAsNumPy(mol)
        
        desc = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), 
                          Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), 
                          Descriptors.NumRotatableBonds(mol), Descriptors.TPSA(mol)])
        
        combined = np.concatenate([fp_morgan, fp_atom_pair, fp_torsion, desc])
        feats.append(combined)
    return np.array(feats, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03, 
                           subsample=0.8, colsample_bytree=0.5, 
                           min_child_weight=3, random_state=seed, 
                           n_jobs=-1, tree_method='hist')
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    eps = 1e-6
    y_train_log = np.log10(y_train + eps)
    model.fit(X_train, y_train_log)
    return model.predict(featurize(test_smiles))
