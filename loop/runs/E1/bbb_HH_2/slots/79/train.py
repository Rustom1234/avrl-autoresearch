"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import rankdata
from sklearn.svm import SVC
from xgboost import XGBClassifier

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


WL_BUCKETS = 4096
WL_ITERS = 3


def wl_counts(smiles):
    """Weisfeiler-Lehman subtree count vectors (initial labels plus WL_ITERS refinements)."""
    x = np.zeros((len(smiles), WL_BUCKETS))
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        atoms = list(m.GetAtoms())
        nbrs = [[n.GetIdx() for n in a.GetNeighbors()] for a in atoms]
        labels = [hash((a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(), int(a.GetIsAromatic()),
                        a.GetTotalNumHs())) for a in atoms]
        for it in range(WL_ITERS + 1):
            for lab in labels:
                x[i, lab % WL_BUCKETS] += 1
            labels = [hash((labels[k], tuple(sorted(labels[j] for j in nbrs[k])))) for k in range(len(atoms))]
    return x


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.int64)
    train_sets = fingerprints(train_smiles)
    test_sets = fingerprints(test_smiles)
    total = np.zeros(len(test_smiles))
    for x_tr, x_te in zip(train_sets, test_sets):
        model = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                              min_child_weight=2, eval_metric='logloss', random_state=seed, n_jobs=8)
        model.fit(x_tr, y)
        total += rankdata(model.predict_proba(x_te)[:, 1])
    # Weisfeiler-Lehman subtree kernel SVM member (cosine-normalised linear kernel)
    wl_tr = wl_counts(train_smiles)
    wl_te = wl_counts(test_smiles)
    wl_tr = wl_tr / (np.linalg.norm(wl_tr, axis=1, keepdims=True) + 1e-12)
    wl_te = wl_te / (np.linalg.norm(wl_te, axis=1, keepdims=True) + 1e-12)
    svc = SVC(kernel='precomputed', C=1.0)
    svc.fit(wl_tr @ wl_tr.T, y)
    total += rankdata(svc.decision_function(wl_te @ wl_tr.T))
    total = (total - total.min()) / (total.max() - total.min() + 1e-12)
    return total
