"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors, AllChem
from catboost import CatBoostRegressor

# Cache for 3D descriptors to speed up repeated fits
_shape_cache = {}

# Settings
N_BITS = 2048
RADIUS = 2
ITERATIONS = 1500
LEARNING_RATE = 0.05
DEPTH = 6


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        
        # Physchem descriptors
        descs = [
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NumAromaticRings(mol)
        ]
        
        # 3D shape descriptors
        canon_smiles = Chem.MolToSmiles(mol)
        if canon_smiles not in _shape_cache:
            try:
                m = Chem.AddHs(mol)
                AllChem.EmbedMolecule(m, AllChem.ETKDGv3, params=AllChem.ETKDGv3.Parameters(randomSeed=0))
                # MMFF optimization is skipped to stay within time limits, but we use the embedded conformer
                # descriptors: PBF, PMI1, PMI2, PMI3, NPR1, NPR2, RadiusOfGyration, Asphericity
                shape_descs = [
                    Descriptors.CalcPBF(m),
                    Descriptors.CalcPMI1(m),
                    Descriptors.CalcPMI2(m),
                    Descriptors.CalcPMI3(m),
                    Descriptors.CalcNPR1(m),
                    Descriptors.CalcNPR2(m),
                    Descriptors.CalcRadiusOfGyration(m),
                    Descriptors.CalcAsphericity(m)
                ]
                _shape_cache[canon_smiles] = np.array(shape_descs, dtype=np.float32)
            except: 
                _shape_cache[canon_smiles] = np.zeros(8, dtype=np.float32)
        
        shape_descs = _shape_cache[canon_smiles]
        fps.append(np.concatenate([fp, np.array(descs, dtype=np.float32), shape_descs]))
    
    return np.array(fps, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    
    model = CatBoostRegressor(
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        depth=DEPTH,
        loss_function='MAE',
        random_seed=seed,
        thread_count=8,
        verbose=0
    )
    model.fit(X_train, y_train)
    
    X_test = featurize(test_smiles)
    return model.predict(X_test)
