"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Settings
C = 3.0
EPSILON = 0.1

def featurize(smiles):
    desc_list = Descriptors.descList
    X = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        features = []
        for name in desc_list:
            try:
                val = getattr(mol, name)() if hasattr(mol, name) else Descriptors.Calc2DDescrByName(mol, name)
                features.append(val)
            except:
                features.append(0.0)
        X.append(features)
    X = np.array(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def fit_predict(train_smiles, train_y, test_smiles, seed):
    y_train = np.asarray(train_y, dtype=np.float64)
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVR(kernel='rbf', C=C, gamma='scale', epsilon=EPSILON)

    if np.all(y_train >= 0):
        y_train_transformed = np.log1p(y_train)
        model.fit(X_train_scaled, y_train_transformed)
        preds = model.predict(X_test_scaled)
        return np.expm1(preds)
    else:
        model.fit(X_train_scaled, y_train)
        return model.predict(X_test_scaled)
