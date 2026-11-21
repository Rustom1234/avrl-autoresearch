"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import rankdata
from xgboost import XGBClassifier
from sklearn.neural_network import MLPRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


ACID = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
BASIC_AMINE = Chem.MolFromSmarts('[NX3;!$(N-C=[O,S,N]);!$(N-a);!$(N-S(=O)=O)]')


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([
            rdMolDescriptors.CalcTPSA(m),
            Descriptors.MolWt(m),
            Descriptors.MolLogP(m),
            rdMolDescriptors.CalcNumHBD(m),
            rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumRotatableBonds(m),
            rdMolDescriptors.CalcFractionCSP3(m),
            Chem.GetFormalCharge(m),
            len(m.GetSubstructMatches(ACID)),
            len(m.GetSubstructMatches(BASIC_AMINE)),
        ])
    return np.array(rows, dtype=np.float32)


def fingerprints(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    pair = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)
    x_morgan = np.array([morgan.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    x_pair = np.array([pair.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    x_maccs = np.array([np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32) for m in mols])
    return [x_morgan, x_pair, x_maccs]


def autoencoder_features(all_smiles, seed):
    # Unsupervised: no labels. Fit an autoencoder on Morgan + MACCS bits, return the 32-unit bottleneck.
    mols = [Chem.MolFromSmiles(s) for s in all_smiles]
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    x = np.array([np.concatenate([morgan.GetFingerprintAsNumPy(m), np.array(list(MACCSkeys.GenMACCSKeys(m)))])
                  for m in mols], dtype=np.float32)
    ae = MLPRegressor(hidden_layer_sizes=(256, 32, 256), activation='relu', max_iter=200, random_state=seed)
    ae.fit(x, x)
    h = x
    for i in range(2):  # encoder: input -> 256 -> 32, ReLU
        h = np.maximum(h @ ae.coefs_[i] + ae.intercepts_[i], 0)
    return h


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.int64)
    train_sets = fingerprints(train_smiles)
    test_sets = fingerprints(test_smiles)
    # Transductive autoencoder member: test features enter only the unsupervised autoencoder fit.
    all_smiles = list(train_smiles) + list(test_smiles)
    n_tr = len(train_smiles)
    x_ae = np.hstack([autoencoder_features(all_smiles, seed), featurize(all_smiles)])
    train_sets.append(x_ae[:n_tr])
    test_sets.append(x_ae[n_tr:])
    total = np.zeros(len(test_smiles))
    for x_tr, x_te in zip(train_sets, test_sets):
        model = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                              min_child_weight=2, eval_metric='logloss', random_state=seed, n_jobs=8)
        model.fit(x_tr, y)
        total += rankdata(model.predict_proba(x_te)[:, 1])
    total = (total - total.min()) / (total.max() - total.min() + 1e-12)
    return total
