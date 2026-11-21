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
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    macc_gen = MACCSkeys.GenMACCSKeys
    smarts_patterns = [
        '[CX3](=O)[OX2H1]',  # Carboxylic acid
        '[NX3;H1,H2,H3]',     # Primary/secondary amine
        '[NX4;](=O)',         # Quaternary nitrogen
        '[OX2H1][P+31]',       # Phosphate
        '[OX2H1][S+61]',       # Sulfonic acid
        '[OX2H1]',             # Hydroxyl
        '[NX3;H1,H2,H3][CX3](=O)', # Amide
    ]
    smarts_mols = [Chem.MolFromSmarts(p) for p in smarts_patterns]
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            morgan = gen.GetCountFingerprintAsNumPy(mol)
            maccs = macc_gen(mol)
            smarts_counts = np.array([int(mol.GetSubstructMatches(sm)[0].NumberOfMatches()) for sm in smarts_mols], dtype=np.float32)
            feats.append(np.concatenate([morgan, np.array(maccs, dtype=np.float32), smarts_counts]))
    return np.array(feats, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = RandomForestClassifier(n_estimators=1000, max_features='sqrt', min_samples_leaf=2, 
                                  class_weight='balanced_subsample', n_jobs=-1, random_state=seed)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(featurize(test_smiles))[:, 1]
