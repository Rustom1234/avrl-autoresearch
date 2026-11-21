"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from xgboost import XGBRegressor

# Settings
N_BITS = 1024
N_ESTIMATORS = 700
LEARNING_RATE = 0.03

def featurize(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    # Atom-pair fingerprints
    ap_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)
    ap_fps = np.array([ap_gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    
    # Topological-torsion count fingerprints
    tr_gen = rdFingerprintGenerator.GetTopologicalTorsionCountGenerator(fpSize=N_BITS)
    tr_fps = np.array([tr_gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    
    # RDKit descriptors
    desc = np.array([[Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.NumHDonors(m), 
                               Descriptors.NumHAcceptors(m), Descriptors.TPSA(m)] 
                      for m in mols], dtype=np.float32)
    
    return np.hstack([ap_fps, tr_fps, desc])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = XGBRegressor(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, 
                           max_depth=6, subsample=0.8, colsample_bytree=0.5, 
                           objective='reg:absoluteerror', n_jobs=-1, random_state=seed)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles))
