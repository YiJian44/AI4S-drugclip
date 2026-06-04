"""DrugCLIP utilities - IO and fingerprint functions."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import DataStructs

def extract_smiles_from_mol2(path):
    """Extract SMILES from Tripos mol2 file - tries SMILES field first, then RDKit."""
    # Try SMILES line first
    try:
        with open(path) as f:
            for line in f:
                if 'smiles' in line.lower() and not line.strip().startswith('#'):
                    for p in line.strip().split():
                        if p not in ('smiles','SMILES','') and not p.startswith('@'):
                            mol = Chem.MolFromSmiles(p)
                            if mol: return p
    except: pass
    # Fallback: let RDKit parse mol2 directly and canonicalize SMILES
    try:
        mol = Chem.MolFromMol2File(str(path))
        if mol:
            return Chem.MolToSmiles(mol)
    except: pass
    return None

def mol_to_fp(mol, dims=[2048, 167, 4096]):
    """Convert RDKit mol to multi-channel fingerprint."""
    if mol is None: return None
    try:
        n2, nc, n3 = dims[0], dims[1], dims[2]
        m2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n2)
        mc = MACCSkeys.GenMACCSKeys(mol)
        m3 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=n3)
        v2 = np.zeros(n2, dtype=np.float32); DataStructs.ConvertToNumpyArray(m2, v2)
        vc = np.zeros(nc, dtype=np.float32); DataStructs.ConvertToNumpyArray(mc, vc)
        v3 = np.zeros(n3, dtype=np.float32); DataStructs.ConvertToNumpyArray(m3, v3)
        return np.concatenate([v2, vc, v3])
    except: return None

def rrf_fusion(score_lists, k=60):
    """Reciprocal Rank Fusion."""
    n = len(score_lists[0])
    rrf = np.zeros(n)
    for scores in score_lists:
        ranks = np.zeros(n, dtype=int)
        ranks[np.argsort(scores)[::-1]] = np.arange(1, n + 1)
        rrf += 1.0 / (ranks + k)
    return rrf

def weighted_sum(score_lists, weights):
    """Weighted sum fusion."""
    ws = np.zeros_like(score_lists[0])
    for s, w in zip(score_lists, weights):
        ws += s * w
    return ws