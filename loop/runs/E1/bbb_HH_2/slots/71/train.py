"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import rankdata
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

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


def minmax_kernel(a, b):
    # Generalised Jaccard: sum_i min / sum_i max, with min(a,b) = sum_t [a>=t][b>=t] over thresholds t.
    a = a.astype(np.int32)
    b = b.astype(np.int32)
    top = int(max(a.max(), b.max()))
    smin = np.zeros((a.shape[0], b.shape[0]))
    for t in range(1, top + 1):
        at = sparse.csr_matrix((a >= t).astype(np.float32))
        bt = sparse.csr_matrix((b >= t).astype(np.float32))
        smin += (at @ bt.T).toarray()
    sa = a.sum(axis=1)[:, None]
    sb = b.sum(axis=1)[None, :]
    return smin / np.maximum(sa + sb - smin, 1e-12)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64) * 2 - 1
    x_tr = fingerprints(train_smiles)[1]
    x_te = fingerprints(test_smiles)[1]
    k = minmax_kernel(x_tr, x_tr)
    k[np.diag_indices_from(k)] += 1.0
    alpha = cho_solve(cho_factor(k), y)
    scores = minmax_kernel(x_te, x_tr) @ alpha
    ranks = rankdata(scores)
    return (ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-12)
