"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors, Descriptors, Crippen
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def descriptors(mol):
    charge = sum(abs(a.GetFormalCharge()) for a in mol.GetAtoms())
    return [Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumRings(mol), mol.GetNumHeavyAtoms(),
            rdMolDescriptors.CalcFractionCSP3(mol), charge]


IONIZATION_PATTERNS = [
    Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]"),                                          # carboxylic acid
    Chem.MolFromSmarts("[#7]1~[#7]~[#7]~[#7]~[#6]~1"),                                    # tetrazole
    Chem.MolFromSmarts("[NX3;H2;!a;!$(NC=[O,S,N]);!$(NS=O)][CX4]"),                       # primary amine
    Chem.MolFromSmarts("[NX3;H1;!a;!$(NC=[O,S,N]);!$(NS=O)]([CX4])[CX4]"),                # secondary amine
    Chem.MolFromSmarts("[NX3;H0;!a;!$(NC=[O,S,N]);!$(NS=O)]([CX4])([CX4])[CX4]"),         # tertiary amine
    Chem.MolFromSmarts("[NX4+]"),                                                         # quaternary ammonium
    Chem.MolFromSmarts("[SX4](=O)(=O)[NX3]"),                                             # sulfonamide
]


def ionization(mol):
    return [len(mol.GetSubstructMatches(p)) for p in IONIZATION_PATTERNS]


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        rows.append(np.concatenate([gen.GetFingerprintAsNumPy(mol),
                                    np.array(descriptors(mol) + ionization(mol), dtype=np.float32)]))
    return np.array(rows, dtype=np.float32)


def scaled_rank(p):
    return (rankdata(p) - 1) / max(len(p) - 1, 1)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    rf = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                n_jobs=-1, random_state=seed)
    rf.fit(X_train, y)
    gb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, colsample_bytree=0.3,
                        random_state=seed, n_jobs=8, verbose=-1)
    gb.fit(X_train, y)
    return 0.5 * (scaled_rank(rf.predict_proba(X_test)[:, 1]) + scaled_rank(gb.predict_proba(X_test)[:, 1]))
