"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, Descriptors
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

# Settings
N_BITS = 2048
RADIUS = 2

def featurize(smiles, include_physchem=False):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    macc_gen = MACCSkeys.GenMACCSKeys
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            morgan = gen.GetCountFingerprintAsNumPy(mol)
            maccs = macc_gen(mol)
            base_feats = np.concatenate([morgan, np.array(maccs, dtype=np.float32)])
            if include_physchem:
                phys = np.array([
                    Descriptors.MolWt(mol), 
                    Descriptors.TPSA(mol), 
                    Descriptors.MolLogP(mol), 
                    Descriptors.NumHDonors(mol), 
                    Descriptors.NumHAcceptors(mol)
                ], dtype=np.float32)
                feats.append(np.concatenate([base_feats, phys]))
            else:
                feats.append(base_feats)
    return np.array(feats, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.int64)
    x_train_base = featurize(train_smiles, include_physchem=False)
    x_test_base = featurize(test_smiles, include_physchem=False)
    x_train_full = featurize(train_smiles, include_physchem=True)
    x_test_full = featurize(test_smiles, include_physchem=True)

    # Random Forest on Morgan bits
    rf = RandomForestClassifier(n_estimators=1000, max_features='sqrt', min_samples_leaf=2, 
                                 class_weight='balanced_subsample', n_jobs=-1, random_state=seed)
    rf.fit(x_train_base, y)
    p_rf = rf.predict_proba(x_test_base)[:, 1]

    # LightGBM on Morgan bits + PhysChem
    train_data = lgb.Dataset(x_train_full, label=y)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.03,
        'colsample_bytree': 0.3,
        'subsample': 0.8,
        'verbosity': -1,
        'seed': seed
    }
    gbm = lgb.train(params, train_data, num_boost_round=500)
    p_gbm = gbm.predict(x_test_full)

    # Rank averaging
    def get_ranks(probs):
        return (np.argsort(np.argsort(probs)) + 1) / len(probs)

    return (get_ranks(p_rf) + get_ranks(p_gbm)) / 2
