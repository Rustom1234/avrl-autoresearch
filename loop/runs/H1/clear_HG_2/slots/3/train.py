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
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        desc = np.array([Descriptors.MolLogP(mol), Descriptors.TPSA(mol), Descriptors.MolWt(mol), 
                          Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), 
                          Descriptors.NumRotatableBonds(mol), Descriptors.FractionCSP3(mol), 
                          Descriptors.NumAromaticRings(mol), Descriptors.HeavyAtomCount(mol), 
                          Descriptors.NumHeteroatoms(mol)])
        feats.append(np.concatenate([fp, desc]))
    return np.array(feats, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03, 
                           subsample=0.8, colsample_bytree=0.5, 
                           min_child_weight=3, random_state=seed, 
                           n_jobs=-1, tree_method='hist')
    X_train = featurize(train_smiles)
    model.fit(X_train, np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles))
