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


def ring_features(mol):
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    parent = list(range(len(rings)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if rings[i] & rings[j]:
                parent[find(i)] = find(j)
    systems = {}
    for i in range(len(rings)):
        systems.setdefault(find(i), []).append(i)
    n_sys = len(systems)
    max_rings = max((len(v) for v in systems.values()), default=0)
    max_atoms = max((len(set().union(*[rings[i] for i in v])) for v in systems.values()), default=0)
    fused_bonds = sum(1 for b in mol.GetBonds() if ri.NumBondRings(b.GetIdx()) >= 2)
    bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    hist = [0] * 7
    for r in rings:
        if len(r) >= 3:
            hist[min(len(r), 9) - 3] += 1
    aromatic = [all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r) for r in rings]
    het_arom = sum(1 for r, ar in zip(rings, aromatic)
                   if ar and any(mol.GetAtomWithIdx(a).GetSymbol() != 'C' for a in r))
    no_arom = sum(1 for r, ar in zip(rings, aromatic)
                  if not ar and any(mol.GetAtomWithIdx(a).GetSymbol() in ('N', 'O') for a in r))
    return [n_sys, max_rings, max_atoms, fused_bonds, bridgeheads] + hist + [het_arom, no_arom]


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        rows.append(np.concatenate([gen.GetFingerprintAsNumPy(mol),
                                    np.array(descriptors(mol) + ring_features(mol), dtype=np.float32)]))
    return np.array(rows, dtype=np.float32)


_DESC_CACHE = {}


def full_descriptors(smiles_list):
    out = []
    for s in smiles_list:
        if s not in _DESC_CACHE:
            mol = Chem.MolFromSmiles(s)
            _DESC_CACHE[s] = [f(mol) for _, f in Descriptors.descList]
        out.append(_DESC_CACHE[s])
    arr = np.array(out, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def scaled_rank(p):
    return (rankdata(p) - 1) / max(len(p) - 1, 1)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    D_train = full_descriptors(train_smiles)
    D_test = full_descriptors(test_smiles)
    lo, hi = np.percentile(D_train, [1, 99], axis=0)
    X_train = np.hstack([X_train, np.clip(D_train, lo, hi).astype(np.float32)])
    X_test = np.hstack([X_test, np.clip(D_test, lo, hi).astype(np.float32)])
    y = np.asarray(train_y, dtype=np.int64)
    rf = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                n_jobs=-1, random_state=seed)
    rf.fit(X_train, y)
    gb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, colsample_bytree=0.3,
                        random_state=seed, n_jobs=8, verbose=-1)
    gb.fit(X_train, y)
    return 0.5 * (scaled_rank(rf.predict_proba(X_test)[:, 1]) + scaled_rank(gb.predict_proba(X_test)[:, 1]))
