"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 'sqrt'
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    props = rdMolDescriptors.Properties()
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        desc = np.array(props.ComputeProperties(mol), dtype=np.float32)
        rows.append(np.concatenate([fp, desc]))
    return np.nan_to_num(np.array(rows, dtype=np.float32), posinf=0.0, neginf=0.0)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y = np.log1p(np.asarray(train_y, dtype=np.float64))
    # Step schedule: learning rate 0.005, halved every 500 rounds, 2000 rounds total.
    lr_schedule = [0.005 * 0.5 ** (i // 500) for i in range(2000)]
    params = dict(objective='regression', learning_rate=0.005, num_leaves=31, min_data_in_leaf=5,
                  feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, seed=seed,
                  num_threads=8, verbose=-1)
    booster = lgb.train(params, lgb.Dataset(X_train, label=y), num_boost_round=2000,
                        callbacks=[lgb.reset_parameter(learning_rate=lr_schedule)])
    pred = booster.predict(X_test)
    return np.expm1(pred)
