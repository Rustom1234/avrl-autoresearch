"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from sklearn.ensemble import RandomForestClassifier

# Settings
N_BITS = 2048
RADIUS = 2


def featurize(smiles):
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    atom_pair_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)
    topo_torsion_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=N_BITS)
    macc_gen = MACCSkeys.GenMACCSKeys
    
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            morgan = morgan_gen.GetCountFingerprintAsNumPy(mol)
            ap = atom_pair_gen.GetCountFingerprintAsNumPy(mol)
            tt = topo_torsion_gen.GetCountFingerprintAsNumPy(mol)
            maccs = macc_gen(mol)
            
            # Concatenate all fingerprints
            feats.append(np.concatenate([morgan, ap, tt, np.array(maccs, dtype=np.float32)]))
    return np.array(feats, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = RandomForestClassifier(n_estimators=1000, max_features='sqrt', min_samples_leaf=2, 
                                  class_weight='balanced_subsample', n_jobs=-1, random_state=seed)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(featurize(test_smiles))[:, 1]
