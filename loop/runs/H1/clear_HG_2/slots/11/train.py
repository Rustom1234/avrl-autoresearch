"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors
import re

# Settings
N_BITS = 2048
RADIUS = 2

# SMARTS for metabolic soft spots
SOFT_SPOTS = {
    'o_methyl': '[OX2M]1C', 
    'n_methyl': '[NX3M]1C', 
    'benzylic_ch2': '[c1;H1]1C\(H2\)', 
    'benzylic_ch3': '[c1;H1]1C(H)(H)', 
    'ar_ch': '[c1;H1]', 
    'tert_aliph': '[C;H3](C)(C)C', 
    'thioether': '[S;H0]([C;H1])', 
    'n_dealkyl': '[NX3;H2]([C;H1])'
}
PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in SOFT_SPOTS.items()}

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
        
        soft_spot_counts = []
        for pat in PATTERNS.values():
            soft_spot_counts.append(float(mol.GetNumMatches(pat)))
        
        ar_ch = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C' and atom.GetIsAromatic() and atom.GetDegree() > 0 and atom.GetPrimeHydrogenCount() > 0)
        ar_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic() and atom.GetSymbol() == 'C')
        ar_ratio = ar_ch / ar_atoms if ar_atoms > 0 else 0.0
        soft_spot_counts.append(ar_ratio)
        
        feats.append(np.concatenate([fp, desc, np.array(soft_spot_counts, dtype=np.float32)]))
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
