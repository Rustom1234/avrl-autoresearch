"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import Data
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from rdkit.Chem import Descriptors, rdMolDescriptors
import lightgbm as lgb

# Settings
# We focus on a descriptor block for monotone constraints

def featurize(smiles):
    feats = []
    acid_smarts = "[CX3](=[OX2H1,OX1-])"
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            # Descriptors
            mw = Descriptors.MolWt(mol)
            tpsa = Descriptors.TPSA(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = rdMolDescriptors.NumHDonors(mol)
            hba = rdMolDescriptors.NumHAcceptors(mol)
            rotb = rdMolDescriptors.NumRotatableBonds(mol)
            charge = abs(sum(atom.GetProp('_charge') if atom.HasProp('_charge') else                                    0 for atom in mol.GetAtoms()))
            
            # Count carboxylic acids
            acid_smarts_mol = Chem.MolFromSmarts(acid_smarts)
            matches = mol.GetSubstructMatches(acid_smarts_mol)
            acid_count = len(matches) if matches else 0
            
            feats.append([mw, tpsa, logp, hbd, hba, rotb, acid_count, charge])
    return np.array(feats, dtype=np.float32)

def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    
    # Features: [MW, TPSA, LogP, HBD, HBA, RotB, Acid, Charge]
    # Constraints: MW(0), TPSA(1), HBD(3), Acid(6), Charge(7) are non-increasing
    # Index mapping for monotone_constraints: 
    # -1 is non-increasing, 1 is non-decreasing
    monotone_constraints = [-1, -1, None, -1, None, None, -1, -1]
    
    train_data = lgb.Dataset(X_train, label=np.asarray(train_y, dtype=np.int64))
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.03,
        'num_leaves': 8,
        'min_child_samples': 20,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'random_state': seed,
        'verbosity': -1
    }
    
    # Use the specific constraints
    # Note: The prompt asks for specific behavior. Let's re-read:
    # TPSA, HBD, MW, Acid, Charge non-increasing -> -1
    # LogP unconstrained -> None
    
    # Re-mapping indices: 
    # 0:MW, 1:TPSA, 2:LogP, 3:HBD, 4:HBA, 5:RotB, 6:Acid, 7:Charge
    
    model = lgb.train({
        **params, 
        'monotone_constraints': monotone_constraints
    }, train_data, 
    num_boost_round=400, 
    valid_sets=[train_data])
    
    return model.predict(X_test)
