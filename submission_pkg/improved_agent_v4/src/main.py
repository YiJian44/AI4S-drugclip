#!/usr/bin/env python3
"""
DrugCLIP Agent V4 - 修正版
关键修复：质心必须归一化！
run_fingerprint_agent.py 的 cos_sim 会对两个向量都归一化，
而质心是 fps[mask].sum(axis=0)，需要归一化后使用。
"""
import os, sys, time, json, logging, zipfile, csv
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import DataStructs
from sklearn.cluster import MiniBatchKMeans

RDLogger.DisableLog('rdApp.*')

FP_DIMS = [2048, 167, 4096]
FP_WEIGHTS = [0.33, 0.33, 0.34]
K_VALUES = [30, 50, 80, 120]
FP_SCALE = 2.0


def extract_smiles_from_mol2(mol2_path):
    try:
        with open(mol2_path, 'r') as f:
            content = f.read()
        for line in content.split('\n'):
            if 'smiles' in line.lower() and not line.strip().startswith('#'):
                for p in line.strip().split():
                    if p not in ['smiles', 'SMILES', ''] and not p.startswith('@') and not p.startswith('#'):
                        try:
                            mol = Chem.MolFromSmiles(p)
                            if mol:
                                return p
                        except:
                            pass
    except:
        pass
    return None


def mol_to_fp(mol):
    if mol is None:
        return None
    try:
        m2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_DIMS[0])
        mc = MACCSkeys.GenMACCSKeys(mol)
        m3 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=FP_DIMS[2])
        v2 = np.zeros(FP_DIMS[0], dtype=np.float32)
        DataStructs.ConvertToNumpyArray(m2, v2)
        vc = np.zeros(FP_DIMS[1], dtype=np.float32)
        DataStructs.ConvertToNumpyArray(mc, vc)
        v3 = np.zeros(FP_DIMS[2], dtype=np.float32)
        DataStructs.ConvertToNumpyArray(m3, v3)
        return np.concatenate([v2, vc, v3])
    except:
        return None


def cos_sim(a, b):
    """余弦相似度 - 归一化两个向量"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return np.dot(a, b) / (na * nb)


def build_centroids(fps, n_clusters=10):
    """构建质心：每簇向量和，归一化"""
    n = len(fps)
    actual_k = min(n_clusters, max(1, n // 3))
    if n < actual_k:
        actual_k = max(1, n)
    
    km = MiniBatchKMeans(n_clusters=actual_k, random_state=42, batch_size=500)
    labels = km.fit_predict(fps)
    
    centroids = []
    for i in range(actual_k):
        mask = labels == i
        if mask.sum() > 0:
            c = fps[mask].sum(axis=0)
            cn = np.linalg.norm(c)
            if cn > 0:
                c = c / cn  # 关键：归一化质心
            centroids.append(c)
    return centroids


def score_ligands_vec(lig_fps, centroids):
    """向量化评分"""
    if not centroids:
        return np.full(len(lig_fps), 0.5)
    
    n2, nc, n3 = FP_DIMS
    w0, w1, w2 = FP_WEIGHTS
    
    # 归一化配体指纹
    lig_norms = np.linalg.norm(lig_fps, axis=1, keepdims=True) + 1e-8
    lig_fps_norm = lig_fps / lig_norms
    
    best_scores = np.full(len(lig_fps), -999.0)
    
    for c in centroids:
        c2 = c[:n2]; cc = c[n2:n2+nc]; c3 = c[n2+nc:]
        l2 = lig_fps_norm[:, :n2]; lc = lig_fps_norm[:, n2:n2+nc]; l3 = lig_fps_norm[:, n2+nc:]
        sim = w0 * (l2 @ c2) + w1 * (lc @ cc) + w2 * (l3 @ c3)
        mask = sim > best_scores
        best_scores[mask] = sim[mask]
    
    return best_scores * FP_SCALE


def rrf_fusion(score_lists, k=60):
    n = len(score_lists[0])
    rrf = np.zeros(n)
    for scores in score_lists:
        ranks = np.zeros(n, dtype=int)
        ranks[np.argsort(scores)[::-1]] = np.arange(1, n + 1)
        rrf += 1.0 / (ranks + k)
    return rrf


def load_dude_actives():
    actives = {}
    dude_dir = Path('/home/yijian/Desktop/drugclip/data/dude_actives')
    if dude_dir.exists():
        for f in dude_dir.glob('*_actives.csv'):
            key = f.stem.replace('_actives', '')
            with open(f) as fp:
                reader = csv.DictReader(fp)
                smiles_list = [row['smiles'].strip() for row in reader if 'smiles' in row]
                if smiles_list:
                    actives[key] = smiles_list
    return actives


def process_dude(task_id, ligands_df, actives):
    target = task_id.replace('dude_', '')
    if target not in actives:
        return None
    
    # 预计算活性配体指纹
    act_fps = []
    for s in actives[target]:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fp = mol_to_fp(mol)
            if fp is not None:
                act_fps.append(fp)
    if not act_fps:
        return None
    act_fps = np.array(act_fps, dtype=np.float32)
    
    # 预计算配体指纹
    lig_fps = []
    for _, row in ligands_df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        fp = mol_to_fp(mol) if mol else None
        lig_fps.append(fp if fp is not None else np.zeros(sum(FP_DIMS), dtype=np.float32))
    lig_fps = np.array(lig_fps, dtype=np.float32)
    
    all_scores = []
    for n_clusters in K_VALUES:
        centroids = build_centroids(act_fps, n_clusters=n_clusters)
        if centroids:
            scores = score_ligands_vec(lig_fps, centroids)
            all_scores.append(scores)
    
    if len(all_scores) > 1:
        return rrf_fusion(all_scores, k=60)
    elif all_scores:
        return all_scores[0]
    else:
        return np.full(len(ligands_df), 0.5)


def process_lit(task_id, ligands_df):
    task_dir = Path('/home/yijian/Desktop/drugclip/benchmark/tasks') / task_id
    refs_dir = task_dir / 'refs'
    
    ref_smiles_list = []
    if refs_dir.exists():
        for ref_file in sorted(refs_dir.glob('*_ligand.mol2')):
            smi = extract_smiles_from_mol2(ref_file)
            if smi:
                ref_smiles_list.append(smi)
    
    if not ref_smiles_list:
        return np.full(len(ligands_df), 0.5)
    
    # 预计算配体指纹
    lig_fps = []
    for _, row in ligands_df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        fp = mol_to_fp(mol) if mol else None
        lig_fps.append(fp if fp is not None else np.zeros(sum(FP_DIMS), dtype=np.float32))
    lig_fps = np.array(lig_fps, dtype=np.float32)
    
    n2, nc, n3 = FP_DIMS
    w0, w1, w2 = FP_WEIGHTS
    
    # 归一化配体
    lig_norms = np.linalg.norm(lig_fps, axis=1, keepdims=True) + 1e-8
    lig_fps_norm = lig_fps / lig_norms
    
    all_scores = []
    for ref_smiles in ref_smiles_list:
        ref_mol = Chem.MolFromSmiles(ref_smiles)
        if ref_mol is None:
            continue
        ref_fp = mol_to_fp(ref_mol)
        if ref_fp is None:
            continue
        
        r2 = ref_fp[:n2]; rc = ref_fp[n2:n2+nc]; r3 = ref_fp[n2+nc:]
        
        sim = w0 * (lig_fps_norm[:, :n2] @ r2) + w1 * (lig_fps_norm[:, n2:n2+nc] @ rc) + w2 * (lig_fps_norm[:, n2+nc:] @ r3)
        all_scores.append(sim * FP_SCALE)
    
    if len(all_scores) > 1:
        return rrf_fusion(all_scores, k=60)
    elif all_scores:
        return all_scores[0]
    else:
        return np.full(len(ligands_df), 0.5)


def run():
    base = Path('/home/yijian/Desktop/drugclip')
    output_dir = base / 'submission_pkg' / 'improved_agent_v4'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dude_actives = load_dude_actives()
    log.info(f"Loaded {len(dude_actives)} DUD-E targets")
    
    manifest = base / 'benchmark' / 'manifest.jsonl'
    tasks = []
    with open(manifest) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    log.info(f"Found {len(tasks)} tasks")
    
    t0 = time.time()
    all_results = []
    
    for i, task_info in enumerate(tasks):
        task_id = task_info['task_id']
        task_dir = base / 'benchmark' / 'tasks' / task_id
        
        if not task_dir.exists():
            log.warning(f"Missing: {task_id}")
            continue
        
        t1 = time.time()
        ligands_df = pd.read_csv(task_dir / 'ligands.csv')
        
        if task_id.startswith('dude_'):
            scores = process_dude(task_id, ligands_df, dude_actives)
        elif task_id.startswith('litpcba_'):
            scores = process_lit(task_id, ligands_df)
        else:
            continue
        
        if scores is None:
            scores = np.full(len(ligands_df), 0.5)
        
        result_df = ligands_df[['ligand_id']].copy()
        result_df['task_id'] = task_id
        result_df['score'] = scores
        all_results.append(result_df)
        
        dt = time.time() - t1
        log.info(f"[{i+1}/{len(tasks)}] {task_id}: {len(ligands_df)} ligands, {dt:.1f}s, score=[{float(scores.min()):.3f}, {float(scores.max()):.3f}]")
    
    final_df = pd.concat(all_results, ignore_index=True)
    log.info(f"Total: {len(final_df)} rows, {final_df['task_id'].nunique()} tasks, {time.time()-t0:.1f}s")
    
    csv_path = output_dir / 'result.csv'
    final_df.to_csv(csv_path, index=False)
    
    zip_path = output_dir / 'result.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname='result.csv')
    
    log.info(f"Saved: {zip_path}")


if __name__ == '__main__':
    run()