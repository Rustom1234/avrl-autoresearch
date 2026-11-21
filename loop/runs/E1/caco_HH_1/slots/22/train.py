"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, AllChem, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


_CACHE = {}


def featurize_one(smi):
    if smi not in _CACHE:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
        m = Chem.MolFromSmiles(smi)
        fp = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6).astype(np.float32)
        _CACHE[smi] = np.concatenate([fp, desc])
    return _CACHE[smi]


def featurize(smiles):
    return np.array([featurize_one(s) for s in smiles], dtype=np.float32)


SHAPE_FUNCS = [rdMolDescriptors.CalcPBF, rdMolDescriptors.CalcPMI1, rdMolDescriptors.CalcPMI2,
               rdMolDescriptors.CalcPMI3, rdMolDescriptors.CalcNPR1, rdMolDescriptors.CalcNPR2,
               rdMolDescriptors.CalcRadiusOfGyration, rdMolDescriptors.CalcInertialShapeFactor]
_SHAPE_CACHE = {}


def shape_one(smi, seed):
    key = (smi, seed)
    if key not in _SHAPE_CACHE:
        vals = np.zeros(len(SHAPE_FUNCS), dtype=np.float64)
        try:
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            if AllChem.EmbedMolecule(m, randomSeed=int(seed) % (2 ** 31)) == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(m, maxIters=200)
                except Exception:
                    pass
                vals = np.array([f(m) for f in SHAPE_FUNCS], dtype=np.float64)
        except Exception:
            vals = np.zeros(len(SHAPE_FUNCS), dtype=np.float64)
        vals = np.clip(np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6).astype(np.float32)
        _SHAPE_CACHE[key] = vals
    return _SHAPE_CACHE[key]


def shape(smiles, seed):
    return np.array([shape_one(s, seed) for s in smiles], dtype=np.float32)


def phys_one(smi):
    m = Chem.MolFromSmiles(smi)
    n = max(m.GetNumHeavyAtoms(), 1)
    arom = sum(a.GetIsAromatic() for a in m.GetAtoms()) / n
    vals = [f(m) for f in (Descriptors.TPSA, Descriptors.MolLogP, Descriptors.NumHDonors,
                           Descriptors.NumHAcceptors, Descriptors.MolWt, Descriptors.NumRotatableBonds)]
    return vals + [arom]


def phys(smiles):
    return np.array([phys_one(s) for s in smiles], dtype=np.float64)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    ptr = phys(train_smiles)
    pte = phys(test_smiles)
    scaler = StandardScaler().fit(ptr)
    lin = Ridge(alpha=1.0).fit(scaler.transform(ptr), y)
    base_tr = lin.predict(scaler.transform(ptr))
    base_te = lin.predict(scaler.transform(pte))
    model = XGBRegressor(objective='reg:absoluteerror', tree_method='hist', max_depth=6, learning_rate=0.03,
                         n_estimators=600, subsample=0.8, colsample_bytree=0.3, min_child_weight=3,
                         random_state=seed, n_jobs=8)
    Xtr = np.hstack([featurize(train_smiles), shape(train_smiles, seed)])
    Xte = np.hstack([featurize(test_smiles), shape(test_smiles, seed)])
    model.fit(Xtr, y - base_tr)
    return base_te + model.predict(Xte)
