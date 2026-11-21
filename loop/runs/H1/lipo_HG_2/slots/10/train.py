"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        
        # Physchem descriptors
        descs = [
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NumAromaticRings(mol)
        ]
        fps.append(np.concatenate([fp, np.array(descs, dtype=np.float32)]))
    
    return np.array(fps, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, 
                                  min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=-1, 
                                  random_state=seed, bootstrap=True)
    model.fit(X_train, y_train)
    
    oob_preds = model.oob_prediction_
    residuals = y_train - oob_preds
    
    X_test = featurize(test_smiles)
    base_preds = model.predict(X_test)
    
    # Use the bit vector portion for similarity (the first N_BITS columns)
    X_train_bits = X_train[:, :N_BITS]
    X_test_bits = X_test[:, :N_BITS]
    
    # Simple cosine-like similarity on bits can proxy for Tanimoto
    # To keep it fast, we use NearestNeighbors with euclidean distance on the bits
    nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn.fit(X_train_bits)
    
    distances, indices = nn.kneighbors(X_test_bits)
    
    # Weighted average of residuals
    # Use 1/(1+dist) as a simple weight to prioritize closer neighbors
    weights = 1.0 / (1.0 + distances)
    
    # Calculate weighted average of residuals for each test point
    # shape of indices: (n_test, 5), shape of residuals: (n_train,)
    weighted_res_sum = np.zeros(len(X_test_bits))
    for i in range(len(X_test_bits)):
        idx = indices[i]
        w = weights[i]
        weighted_res_sum[i] = np.sum(residuals[idx] * w) / np.sum(w)
        
    return base_preds + weighted_res_sum
