"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

N_ROT_TREES = 200
N_TOP_BITS = 300
SUBSET_SIZE = 10

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


def rotation_forest(x_tr, y, x_te, seed):
    rng = np.random.RandomState(seed)
    mu = x_tr.mean(axis=0)
    sd = x_tr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    z_tr = (x_tr - mu) / sd
    z_te = (x_te - mu) / sd
    n, n_feat = z_tr.shape
    total = np.zeros(z_te.shape[0])
    for t in range(N_ROT_TREES):
        perm = rng.permutation(n_feat)
        rot_tr, rot_te = [], []
        for start in range(0, n_feat, SUBSET_SIZE):
            cols = perm[start:start + SUBSET_SIZE]
            rows = rng.choice(n, size=int(0.75 * n), replace=False)
            pca = PCA().fit(z_tr[rows][:, cols])
            rot_tr.append(pca.transform(z_tr[:, cols]))
            rot_te.append(pca.transform(z_te[:, cols]))
        tree = DecisionTreeClassifier(max_depth=None, min_samples_leaf=2, random_state=seed + t)
        tree.fit(np.hstack(rot_tr), y)
        total += tree.predict_proba(np.hstack(rot_te))[:, 1]
    return total / N_ROT_TREES


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.int64)
    desc_tr = featurize(train_smiles)
    desc_te = featurize(test_smiles)
    morgan_tr = fingerprints(train_smiles)[0]
    morgan_te = fingerprints(test_smiles)[0]
    top = np.argsort(-morgan_tr.sum(axis=0))[:N_TOP_BITS]
    x_tr = np.hstack([desc_tr, morgan_tr[:, top]])
    x_te = np.hstack([desc_te, morgan_te[:, top]])
    return rotation_forest(x_tr, y, x_te, seed)
