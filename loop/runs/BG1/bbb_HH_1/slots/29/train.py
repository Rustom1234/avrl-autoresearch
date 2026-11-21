"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from lightgbm import LGBMClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def ring_features(mol):
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    systems = []
    for r in rings:
        new = set(r)
        keep = []
        for s in systems:
            if s & new:
                new |= s
            else:
                keep.append(s)
        systems = keep + [new]
    ring_atoms = set().union(*rings) if rings else set()
    n_arom = sum(all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r) for r in rings)
    n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    return [
        len(systems),
        max((len(s) for s in systems), default=0),
        n_arom,
        len(rings) - n_arom,
        rdMolDescriptors.CalcFractionCSP3(mol),
        n_stereo,
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        mol.GetNumHeavyAtoms() / max(len(ring_atoms), 1),
    ]


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fp = np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    desc = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=np.float64)
    desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    ring = np.array([ring_features(m) for m in mols], dtype=np.float32)
    return np.hstack([fp, desc.astype(np.float32), ring])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
                           colsample_bytree=0.3, subsample=0.8, subsample_freq=1, min_child_samples=10,
                                  n_jobs=-1, random_state=seed)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(featurize(test_smiles))[:, 1]
