"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np

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
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y_train = np.asarray(train_y, dtype=np.int64)

    rf_params = {'n_estimators': 1000, 'max_features': 'sqrt', 'min_samples_leaf': 2, 
                   'class_weight': 'balanced_subsample', 'n_jobs': -1, 'random_state': seed}
    lgb_params = {'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 7, 
                    'random_state': seed, 'verbose': -1}

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_rf = np.zeros(len(X_train))
    oof_lgb = np.zeros(len(X_train))

    for train_idx, val_idx in kf.split(X_train, y_train):
        rf = RandomForestClassifier(**rf_params)
        lgb = LGBMClassifier(**lgb_params)
        rf.fit(X_train[train_idx], y_train[train_idx])
        lgb.fit(X_train[train_idx], y_train[train_idx])
        oof_rf[val_idx] = rf.predict_proba(X_train[val_idx])[:, 1]
        oof_lgb[val_idx] = lgb.predict_proba(X_train[val_idx])[:, 1]

    meta_model = LogisticRegression(C=1.0, random_state=seed)
    meta_model.fit(np.column_stack([oof_rf, oof_lgb]), y_train)

    rf_full = RandomForestClassifier(**rf_params)
    lgb_full = LGBMClassifier(**lgb_params)
    rf_full.fit(X_train, y_train)
    lgb_full.fit(X_train, y_train)

    test_rf = rf_full.predict_proba(X_test)[:, 1]
    test_lgb = lgb_full.predict_proba(X_test)[:, 1]
    
    return meta_model.predict_proba(np.column_stack([test_rf, test_lgb]))[:, 1]
