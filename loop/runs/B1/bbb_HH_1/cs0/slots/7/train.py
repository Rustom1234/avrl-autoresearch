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


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        rows.append(np.concatenate([gen.GetFingerprintAsNumPy(mol), np.array(descriptors(mol), dtype=np.float32)]))
    return np.array(rows, dtype=np.float32)


def shape_features(mol):
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.BondRings():
        bs = set(ring)
        merged = [s for s in systems if s[0] & bs]
        for s in merged:
            systems.remove(s)
            bs |= s[0]
        systems.append((bs, None))
    sizes = []
    for bs, _ in systems:
        atoms = set()
        for b in bs:
            bond = mol.GetBondWithIdx(b)
            atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        sizes.append(len(atoms))
    n = mol.GetNumHeavyAtoms()
    chain = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and not a.IsInRing()}
    longest = 0
    for start in chain:
        dist = {start: 1}
        queue = [start]
        for u in queue:
            for nb in mol.GetAtomWithIdx(u).GetNeighbors():
                v = nb.GetIdx()
                if v in chain and v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        longest = max(longest, max(dist.values()))
    branch = sum(1 for a in mol.GetAtoms() if not a.IsInRing() and a.GetDegree() >= 3)
    diameter = Chem.GetDistanceMatrix(mol).max() if n > 1 else 0
    ring_atoms = sum(1 for a in mol.GetAtoms() if a.IsInRing())
    return [max(sizes) if sizes else 0, len(sizes), rdMolDescriptors.CalcNumSpiroAtoms(mol),
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol), longest, branch, diameter,
            ring_atoms / max(n, 1)]


_DESC_CACHE = {}


def full_descriptors(smiles_list):
    out = []
    for s in smiles_list:
        if s not in _DESC_CACHE:
            mol = Chem.MolFromSmiles(s)
            _DESC_CACHE[s] = [f(mol) for _, f in Descriptors.descList] + shape_features(mol)
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
