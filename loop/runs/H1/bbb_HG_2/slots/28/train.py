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
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            morgan = gen.GetCountFingerprintAsNumPy(mol)
            maccs = macc_gen(mol)
            feats.append(np.concatenate([morgan, np.array(maccs, dtype=np.float32)]))
    return np.array(feats, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    x_train = featurize(train_smiles)
    x_test = featurize(test_smiles)
    y_train = np.asarray(train_y, dtype=np.int64)

    # Quick selector to find top 512 features
    selector = RandomForestClassifier(n_estimators=200, max_features='sqrt', min_samples_leaf=2, 
                                      class_weight='balanced', n_jobs=-1, random_state=seed)
    selector.fit(x_train, y_train)
    importances = selector.feature_importances_
    top_indices = np.argsort(importances)[-512:]

    # Train final model on selected features
    model = RandomForestClassifier(n_estimators=1000, max_features='sqrt', min_samples_leaf=2, 
                                  class_weight='balanced_subsample', n_jobs=-1, random_state=seed)
    model.fit(x_train[:, top_indices], y_train)
    return model.predict_proba(x_test[:, top_indices])[:, 1]
