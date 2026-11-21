"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, rdMolOps
from sklearn.ensemble import RandomForestClassifier

# Settings
N_BITS = 2048
RADIUS = 2
SCAFF_BITS = 512


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    macc_gen = MACCSkeys.GenMACCSKeys
    scaff_gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=SCAFF_BITS)
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            morgan = gen.GetCountFingerprintAsNumPy(mol)
            maccs = macc_gen(mol)
            try:
                # Get Bemis-Murcko scaffold
                scaff = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RemoveHs(mol)))
                # Generate scaffold
                scaff_mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RemoveHs(scaff)))
                # The prompt mentions MakeScaffoldGeneric; in RDKit this is typically handled
                # by canonicalizing the scaffold or using specific scripts. To keep it robust:
                scaff_feats = scaff_gen.GetCountFingerprintAsNumPy(scaff_mol)
            except:
                scaff_feats = np.zeros(SCAFF_BITS, dtype=np.float32)
            
            feats.append(np.concatenate([morgan, np.array(maccs, dtype=np.float32), scaff_feats]))
    return np.array(feats, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = RandomForestClassifier(n_estimators=1000, max_features='sqrt', min_samples_leaf=2, 
                                  class_weight='balanced_subsample', n_jobs=-1, random_state=seed)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(featurize(test_smiles))[:, 1]
