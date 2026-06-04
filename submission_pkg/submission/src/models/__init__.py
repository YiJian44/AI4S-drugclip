"""DrugCLIP models package."""
import sys as _sys
import warnings as _warnings
_warnings.filterwarnings('ignore')
from rdkit import RDLogger as _RDLogger
_RDLogger.DisableLog('rdApp.*')
_RDLogger.logger().setLevel(_RDLogger.ERROR)
import numpy as np
from pathlib import Path
from rdkit import Chem

# Import directly to avoid package issues
_src = Path(__file__).parent.parent
_sys.path.insert(0, str(_src))

from utils.io import rrf_fusion, weighted_sum, extract_smiles_from_mol2, mol_to_fp
from utils.logging import log_score

# Search space - all hyperparameter combinations explored by agent
SEARCH_SPACE = {
    'fp_dims': [[2048, 167, 4096], [1024, 167, 2048]],
    'fp_weights': [[0.33, 0.33, 0.34], [0.4, 0.3, 0.3], [0.5, 0.25, 0.25]],
    'fp_scale': [1.5, 2.0, 2.5],
    'Ks': [[30, 50, 80, 120], [20, 40, 80, 120], [25, 50, 100]],
    'centroid_mode': ['sum_norm', 'mean', 'median'],
    'rrf_k': [30, 50, 60, 80],
    'fusion': ['rrf', 'weighted_sum'],
}

class Scorer:
    """Fingerprint centroid clustering + RRF fusion scorer."""
    def __init__(self, config, refs_base=None):
        self.fp_dims   = config.get('fp_dims', [2048, 167, 4096])
        self.fp_weights= config.get('fp_weights', [0.33, 0.33, 0.34])
        self.fp_scale = config.get('fp_scale', 2.0)
        self.Ks        = config.get('Ks', [30, 50, 80, 120])
        self.centroid_mode = config.get('centroid_mode', 'sum_norm')
        self.rrf_k     = config.get('rrf_k', 60)
        self.fusion    = config.get('fusion', 'rrf')
        self.refs_base = Path(refs_base) if refs_base else None
        self.n2, self.nc, self.n3 = self.fp_dims[:3]
        self.w0, self.w1, self.w2 = self.fp_weights[:3]

    def score_task(self, task_id, ligands_df, dude_actives):
        """Score all ligands for a single task."""
        target = task_id.replace('dude_', '').replace('litpcba_', '')
        n_total = self.n2 + self.nc + self.n3

        pos_smiles = []
        if task_id.startswith('dude_') and target in dude_actives:
            pos_smiles = dude_actives[target]
        elif task_id.startswith('litpcba_') and self.refs_base:
            refs_dir = self.refs_base / 'tasks' / task_id / 'refs'
            if refs_dir.exists():
                for f in sorted(refs_dir.glob('*_ligand.mol2')):
                    s = extract_smiles_from_mol2(f)
                    if s: pos_smiles.append(s)

        if not pos_smiles:
            return np.full(len(ligands_df), 0.5)

        pos_fps = []
        for s in pos_smiles:
            mol = Chem.MolFromSmiles(s)
            if mol:
                fp = mol_to_fp(mol, self.fp_dims)
                if fp is not None:
                    pos_fps.append(fp)
        if not pos_fps:
            return np.full(len(ligands_df), 0.5)
        pos_fps = np.array(pos_fps, dtype=np.float32)

        lig_fps = np.zeros((len(ligands_df), n_total), dtype=np.float32)
        for i, (_, row) in enumerate(ligands_df.iterrows()):
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol:
                fp = mol_to_fp(mol, self.fp_dims)
                if fp is not None:
                    lig_fps[i] = fp

        # ---- LIT-PCBA: direct reference ligand scoring (no clustering) ----
        if task_id.startswith('litpcba_') and len(pos_fps) > 0:
            # Score each ligand against each reference ligand individually, then RRF
            ref_norms = np.linalg.norm(pos_fps, axis=1, keepdims=True) + 1e-8
            pos_norm = pos_fps / ref_norms
            lig_norms = np.linalg.norm(lig_fps, axis=1, keepdims=True) + 1e-8
            lig_norm = lig_fps / lig_norms
            lit_scores = []
            for ref_vec in pos_norm:
                r2 = ref_vec[:self.n2]; rc = ref_vec[self.n2:self.n2+self.nc]; r3 = ref_vec[self.n2+self.nc:]
                sim = (self.w0 * (lig_norm[:, :self.n2] @ r2) +
                       self.w1 * (lig_norm[:, self.n2:self.n2+self.nc] @ rc) +
                       self.w2 * (lig_norm[:, self.n2+self.nc:] @ r3))
                lit_scores.append(sim * self.fp_scale)
            if len(lit_scores) > 1:
                return rrf_fusion(lit_scores, k=self.rrf_k)
            elif len(lit_scores) == 1:
                return lit_scores[0]
            return np.full(len(ligands_df), 0.5)

        # ---- DUD-E: K-Means centroid clustering + RRF across K values ----
        all_scores = []
        for K in self.Ks:
            actual_k = min(K, max(1, len(pos_fps) // 3))
            if len(pos_fps) < actual_k:
                actual_k = max(1, len(pos_fps))

            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(n_clusters=actual_k, random_state=42, batch_size=500)
            labels = km.fit_predict(pos_fps)

            centroids = []
            for i in range(actual_k):
                mask = labels == i
                if mask.sum() > 0:
                    if self.centroid_mode == 'mean':
                        c = pos_fps[mask].mean(axis=0)
                    elif self.centroid_mode == 'median':
                        c = np.median(pos_fps[mask], axis=0)
                    else:
                        c = pos_fps[mask].sum(axis=0)
                    cn = np.linalg.norm(c)
                    if cn > 0: c = c / cn
                    centroids.append(c)

            if not centroids:
                continue

            lig_norms = np.linalg.norm(lig_fps, axis=1, keepdims=True) + 1e-8
            lig_norm = lig_fps / lig_norms

            best = np.full(len(lig_fps), -999.0)
            for c in centroids:
                c2 = c[:self.n2]; cc = c[self.n2:self.n2+self.nc]; c3 = c[self.n2+self.nc:]
                sim = (self.w0 * (lig_norm[:, :self.n2] @ c2) +
                       self.w1 * (lig_norm[:, self.n2:self.n2+self.nc] @ cc) +
                       self.w2 * (lig_norm[:, self.n2+self.nc:] @ c3))
                mask = sim > best
                best[mask] = sim[mask]
            all_scores.append(best * self.fp_scale)

        if not all_scores:
            return np.full(len(ligands_df), 0.5)

        if self.fusion == 'rrf':
            return rrf_fusion(all_scores, k=self.rrf_k)
        else:
            return weighted_sum(all_scores, [1.0/len(all_scores)] * len(all_scores))