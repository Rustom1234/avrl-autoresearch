"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from catboost import CatBoostClassifier
from rdkit.Chem import Descriptors

# Settings
N_BITS = 2048
RADIUS = 2

def featurize(smiles):
    gen_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    gen_atom_pair = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024)
    gen_torsion = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024)
    
    features = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        
        fp_morgan = gen_morgan.GetFingerprintAsNumPy(mol)
        fp_atom_pair = gen_atom_pair.GetFingerprintAsNumPy(mol)
        fp_torsion = gen_torsion.GetFingerprintAsNumPy(mol)
        
        # Add basic descriptors: MW, LogP, TPSA
        desc = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol)])
        
        features.append(np.concatenate([fp_morgan, fp_atom_pair, fp_torsion, desc]))
    return np.array(features, dtype=np.float32)

def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=seed,
        verbose=False,
        thread_count=8
    )
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(featurize(test_smiles))[:, 1]
