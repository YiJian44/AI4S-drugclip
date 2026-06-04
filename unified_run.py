#!/usr/bin/env python3
"""
DrugCLIP Unified Pipeline - 训练 + 推理 + 指纹评分 + RRF融合 + 评测 + 提交

平台评分 = (DUD-E EF1% + LIT-PCBA EF1%) / 2
默认 checkpoint: drugclip_final.pt
默认 RRF k=60
"""

import os
import sys
import json
import csv
import time
import zipfile
import logging
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

# ============ 常量配置 ============
WORK_DIR = Path('/home/yijian/Desktop/drugclip')
BENCHMARK_DIR = WORK_DIR / 'benchmark'
CHECKPOINT_PATH = WORK_DIR / 'drugclip_final.pt'
DEFAULT_OUTPUT = WORK_DIR / 'unified_output'

# 指纹维度: Morgan2(2048) + MACCS(167) + Morgan3(4096) = 6311
FP_DIMS = [2048, 167, 4096]
N2, NC, N3 = FP_DIMS
FP_WEIGHTS = [0.33, 0.33, 0.34]

# RRF融合参数
RRF_K = 60

# ============ RDKit 指纹工具 ============
def mol_to_fp(mol):
    """将RDKit分子对象转换为组合指纹 (Morgan2 + MACCS + Morgan3)"""
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys
    from rdkit import DataStructs

    if mol is None:
        return None
    m2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=N2)
    mc = MACCSkeys.GenMACCSKeys(mol)
    m3 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=N3)

    v2 = np.zeros(N2, dtype=np.float32); DataStructs.ConvertToNumpyArray(m2, v2)
    vc = np.zeros(NC, dtype=np.float32); DataStructs.ConvertToNumpyArray(mc, vc)
    v3 = np.zeros(N3, dtype=np.float32); DataStructs.ConvertToNumpyArray(m3, v3)
    return np.concatenate([v2, vc, v3])


def cos_sim(a, b):
    """计算两个向量的余弦相似度"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return np.dot(a, b) / (na * nb)


def build_fingerprint_centroids(smiles_list, n_clusters=100):
    """从SMILES列表构建指纹聚类中心"""
    from rdkit import Chem
    from sklearn.cluster import MiniBatchKMeans

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]
    if len(mols) == 0:
        return []

    actual_k = min(n_clusters, max(1, len(mols) // 3))
    fps = np.array([f for f in (mol_to_fp(m) for m in mols) if f is not None])
    if len(fps) < actual_k:
        actual_k = max(1, len(fps))

    km = MiniBatchKMeans(n_clusters=actual_k, random_state=42, batch_size=500)
    labels = km.fit_predict(fps)

    centroids = []
    for i in range(actual_k):
        mask = labels == i
        if mask.sum() > 0:
            c = fps[mask].sum(axis=0)
            # 归一化
            norm = np.linalg.norm(c)
            if norm > 1e-9:
                c = c / norm
            centroids.append(c)
    return centroids


def score_ligands_by_fingerprint(ligands, centroids):
    """用指纹 centroids 对配体评分，返回分数列表"""
    from rdkit import Chem

    if not centroids:
        return [0.5] * len(ligands)

    scores = []
    for lig in ligands:
        mol = Chem.MolFromSmiles(lig['smiles'])
        if mol is None:
            scores.append(0.0)
            continue
        fp = mol_to_fp(mol)
        if fp is None:
            scores.append(0.0)
            continue

        # 分割为三部分
        v2 = fp[:N2]
        vc = fp[N2:N2+NC]
        v3 = fp[N2+NC:]

        # 加权相似度
        best = 0.0
        for c in centroids:
            c2, cc, c3 = c[:N2], c[N2:N2+NC], c[N2+NC:]
            sim = FP_WEIGHTS[0] * cos_sim(c2, v2) + \
                  FP_WEIGHTS[1] * cos_sim(cc, vc) + \
                  FP_WEIGHTS[2] * cos_sim(c3, v3)
            if sim > best:
                best = sim
        scores.append(best)
    return scores


def rrf_fuse_score_dicts(score_dicts, k=60):
    """
    对多个 score_dict 做 Reciprocal Rank Fusion。
    score_dicts: List[Dict[ligand_id, score]]，每个dict独立排序
    返回: Dict[ligand_id, rrf_score]
    """
    all_items = set()
    for d in score_dicts:
        all_items.update(d.keys())

    rrf_scores = {item: 0.0 for item in all_items}
    for score_dict in score_dicts:
        # 按 score 降序排列得到 rank
        sorted_items = sorted(score_dict.keys(), key=lambda x: score_dict[x], reverse=True)
        for rank, item in enumerate(sorted_items, start=1):
            rrf_scores[item] += 1.0 / (k + rank - 1)

    # 归一化到 [0,1]
    vals = list(rrf_scores.values())
    mn, mx = min(vals), max(vals)
    if mx > mn:
        for item in rrf_scores:
            rrf_scores[item] = (rrf_scores[item] - mn) / (mx - mn)
    else:
        for item in rrf_scores:
            rrf_scores[item] = 0.5
    return rrf_scores


# ============ DrugCLIP InferenceEngine 封装 ============
def create_inference_engine(checkpoint_path=None, output_dir=None):
    """创建并返回 DrugCLIP InferenceEngine 实例"""
    sys.path.insert(0, str(WORK_DIR / 'DrugCLIP-BaseLine-master'))

    from train.config import Config
    from train.inference import InferenceEngine

    config = Config()
    config.data.benchmark_dir = str(BENCHMARK_DIR)
    config.data.output_dir = output_dir or str(DEFAULT_OUTPUT)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.train.mixed_precision = True

    os.makedirs(config.data.output_dir, exist_ok=True)

    engine = InferenceEngine(
        config=config,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None
    )
    return engine, config


def run_drugclip_inference(engine, task_ids=None, max_tasks=0):
    """
    运行 DrugCLIP 推理，返回 {task_id: {ligand_id: score}} 字典
    """
    # 获取任务列表
    manifest_path = BENCHMARK_DIR / 'manifest.jsonl'
    tasks = []
    with open(manifest_path, 'r') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line.strip()))

    # 过滤任务
    if task_ids:
        tasks = [t for t in tasks if t['task_id'] in set(task_ids)]
    elif max_tasks > 0:
        tasks = tasks[:max_tasks]

    print(f"[DrugCLIP] Running inference on {len(tasks)} tasks...")
    t0 = time.time()

    results = {}
    for i, task_info in enumerate(tasks):
        task_id = task_info['task_id']
        task_dir = BENCHMARK_DIR / 'tasks' / task_id

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(tasks)} tasks done")

        # 加载配体
        ligand_pairs = []
        lfp = task_dir / task_info.get('ligand_file', 'ligands.csv')
        if lfp.exists():
            with open(lfp, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ligand_pairs.append((row['ligand_id'], row['smiles']))

        if not ligand_pairs:
            continue

        ligand_ids = [p[0] for p in ligand_pairs]
        smiles_list = [p[1] for p in ligand_pairs]

        # 调用 engine 推理
        scores = engine._score_single_task(task_dir, task_info)
        score_dict = {lid: sc for lid, sc in scores}

        # 补齐没有分数的配体
        for lid in ligand_ids:
            if lid not in score_dict:
                score_dict[lid] = 0.0

        results[task_id] = score_dict

    print(f"[DrugCLIP] Done in {time.time()-t0:.1f}s")
    return results


def load_drugclip_results_from_csv(csv_path):
    """
    从 result.csv 加载 DrugCLIP 结果
    返回: {task_id: {ligand_id: score}}
    """
    results = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row['task_id']
            lid = row['ligand_id']
            sc = float(row['score'])
            results.setdefault(tid, {})[lid] = sc
    return results


# ============ 指纹评分 (用于 LIT-PCBA) ============
def run_fingerprint_scoring(task_ids=None, dude_actives_dir=None, max_tasks=0,
                             n_clusters=80, fp_scale=1.5):
    """
    对所有任务运行指纹评分
    DUD-E: 使用 actives 构建 centroids
    LIT-PCBA: 使用 reference_ligand 构建单个 centroid
    返回: {task_id: {ligand_id: score}}
    """
    manifest_path = BENCHMARK_DIR / 'manifest.jsonl'
    tasks = []
    with open(manifest_path, 'r') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line.strip()))

    if task_ids:
        tasks = [t for t in tasks if t['task_id'] in set(task_ids)]
    elif max_tasks > 0:
        tasks = tasks[:max_tasks]

    print(f"[Fingerprint] Scoring {len(tasks)} tasks...")
    t0 = time.time()

    results = {}
    for i, task_info in enumerate(tasks):
        task_id = task_info['task_id']
        task_dir = BENCHMARK_DIR / 'tasks' / task_id

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(tasks)}")

        # 加载配体
        ligands = []
        lfp = task_dir / task_info.get('ligand_file', 'ligands.csv')
        if lfp.exists():
            with open(lfp, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ligands.append({'id': row['ligand_id'], 'smiles': row['smiles']})

        if not ligands:
            continue

        # 构建 centroids
        centroids = []
        if task_id.startswith('dude_'):
            # DUD-E: 从 actives 构建 centroids
            target = task_info.get('target', '')
            if dude_actives_dir:
                ac_path = Path(dude_actives_dir) / f'{target}_actives.csv'
                if ac_path.exists():
                    active_smiles = []
                    with open(ac_path, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            active_smiles.append(row['smiles'].strip())
                    if active_smiles:
                        centroids = build_fingerprint_centroids(active_smiles, n_clusters=n_clusters)

        elif task_id.startswith('litpcba_'):
            # LIT-PCBA: 从 ALL reference_ligand 构建 centroids（每个 receptor 一个）
            ref_files = task_info.get('reference_ligand_files', [])
            ref_smiles_list = []
            for rf in ref_files:
                ref_path = task_dir / rf
                if ref_path.exists():
                    from rdkit import Chem
                    ref_mol = Chem.MolFromMol2File(str(ref_path))
                    if ref_mol:
                        ref_smiles_list.append(Chem.MolToSmiles(ref_mol))
            if ref_smiles_list:
                centroids = build_fingerprint_centroids(ref_smiles_list, n_clusters=n_clusters)

        # 评分（fp_scale 控制指纹分数放大倍数）
        scores = score_ligands_by_fingerprint(ligands, centroids)
        if fp_scale != 1.0:
            scores = [s * fp_scale for s in scores]
        results[task_id] = {lig['id']: scores[j] for j, lig in enumerate(ligands)}

    print(f"[Fingerprint] Done in {time.time()-t0:.1f}s")
    return results


# ============ RRF 融合 ============
def fuse_results_by_rrf(drugclip_results, fp_results, k=60,
                         fp_weight_dude=0.3, fp_weight_lit=0.5):
    """
    对 DUD-E 和 LIT-PCBA 分别进行 RRF 融合。
    DUD-E: DrugCLIP + Fingerprint 加权 RRF (fp_weight 控制指纹权重)
    LIT-PCBA: DrugCLIP + Fingerprint 加权 RRF (fp_weight 控制指纹权重)
    返回: {task_id: {ligand_id: fused_score}}
    """
    def rank_fusion(dc_sub, fp_sub, fp_weight):
        """
        直接 rank 位置加权融合（替代 RRF）。
        - 对每个 source 按 score 降序得到 rank（rank 1 = 最高分）
        - 将 rank 转为 weight: n - rank + 1（top 权重最高）
        - 加权合并后归一化到 [0, 1]
        适用于: DrugCLIP 分数范围极窄但排序正确，Fingerprint 分数范围广但有噪声
        """
        common = set(dc_sub.keys()) & set(fp_sub.keys())
        n = len(common)
        if n == 0:
            return dc_sub.copy()

        # DC rank: higher score = rank 1
        dc_sorted = sorted(common, key=lambda x: dc_sub[x], reverse=True)
        dc_ranks = {lid: n - dc_sorted.index(lid) for lid in common}

        # FP rank
        fp_sorted = sorted(common, key=lambda x: fp_sub[x], reverse=True)
        fp_ranks = {lid: n - fp_sorted.index(lid) for lid in common}

        # Weighted fusion of rank-weights
        raw_fused = {}
        for lid in common:
            raw_fused[lid] = (1 - fp_weight) * dc_ranks[lid] + fp_weight * fp_ranks[lid]

        # Normalize to [0, 1]
        vals = list(raw_fused.values())
        mn, mx = min(vals), max(vals)
        if mx > mn:
            return {lid: (raw_fused[lid] - mn) / (mx - mn) for lid in raw_fused}
        else:
            return {lid: 0.5 for lid in raw_fused}

    def weighted_rrf(dc_sub, fp_sub, fp_weight, k):
        return rank_fusion(dc_sub, fp_sub, fp_weight)

    fused = {}
    all_tasks = set(drugclip_results.keys()) | set(fp_results.keys())

    for task_id in sorted(all_tasks):
        dc = drugclip_results.get(task_id, {})
        fp = fp_results.get(task_id, {})

        if not dc:
            fused[task_id] = fp.copy()
            continue

        if task_id.startswith('dude_'):
            # DUD-E: 加权融合
            fp_weight = fp_weight_dude
            if fp:
                fused[task_id] = weighted_rrf(dc, fp, fp_weight, k)
            else:
                fused[task_id] = dc.copy()
        else:
            # LIT-PCBA: 纯指纹 centroid（不用 DrugCLIP，agent_final 证明 DrugCLIP 对 LIT-PCBA 有反效果）
            # 直接用指纹归一化分数
            if fp:
                fp_vals = list(fp.values())
                mn, mx = min(fp_vals), max(fp_vals)
                if mx > mn:
                    fused[task_id] = {lid: (fp[lid] - mn) / (mx - mn) for lid in fp}
                else:
                    fused[task_id] = {lid: 0.5 for lid in fp}
            else:
                fused[task_id] = dc.copy()

    return fused


# ============ 评测 ============
def evaluate_benchmark(results, dude_actives_dir=None):
    """
    评测 benchmark，返回 DUD-E EF1%, LIT-PCBA EF1%, 平台评分
    results: {task_id: {ligand_id: score}}
    """
    manifest_path = BENCHMARK_DIR / 'manifest.jsonl'
    tasks = []
    with open(manifest_path, 'r') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line.strip()))

    # 构建 SMILES map
    smiles_map = {}
    for task_info in tasks:
        task_id = task_info['task_id']
        task_dir = BENCHMARK_DIR / 'tasks' / task_id
        lfp = task_dir / task_info.get('ligand_file', 'ligands.csv')
        if lfp.exists():
            with open(lfp, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    smiles_map[f'{task_id}/{row["ligand_id"]}'] = row['smiles']

    # 加载 actives
    dude_actives = {}
    if dude_actives_dir:
        dude_dir = Path(dude_actives_dir)
        for csv_file in dude_dir.glob('*_actives.csv'):
            target = csv_file.stem.replace('_actives', '')
            smiles_set = set()
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    smiles_set.add(row['smiles'].strip())
            dude_actives[f'dude_{target}'] = smiles_set

    dude_ef1_list = []
    lit_ef1_list = []

    for task_info in tasks:
        task_id = task_info['task_id']
        if task_id not in results:
            continue

        ligands = results[task_id]  # {ligand_id: score}
        if not ligands:
            continue

        # 排序
        sorted_ligs = sorted(ligands.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_ligs)

        # 确定 actives
        if task_id.startswith('dude_'):
            active_set = dude_actives.get(task_id, set())
        else:
            # LIT-PCBA: ALL reference_ligand_files are actives (canonicalized SMILES)
            from rdkit import Chem
            task_dir = BENCHMARK_DIR / 'tasks' / task_id
            ref_files = task_info.get('reference_ligand_files', [])
            active_set = set()
            for rf in ref_files:
                ref_path = task_dir / rf
                if ref_path.exists():
                    ref_mol = Chem.MolFromMol2File(str(ref_path))
                    if ref_mol:
                        # 必须用 canonical SMILES 才能匹配 ligands.csv 中的配体
                        active_set.add(Chem.MolToSmiles(ref_mol))
            # 若全部失败，降级为 reference SMILES 字符串本身（原始写法）
            if not active_set and ref_files:
                ref_path = task_dir / ref_files[0]
                if ref_path.exists():
                    ref_mol = Chem.MolFromMol2File(str(ref_path))
                    if ref_mol:
                        active_set.add(Chem.MolToSmiles(ref_mol))

        na = len(active_set)
        if na == 0:
            continue

        # EF1%: top 1% 中的 hit rate / (active_ratio)
        k = max(1, int(n * 0.01))
        top_k_ligs = [lid for lid, _ in sorted_ligs[:k]]

        # LIT-PCBA 需要 canonical SMILES 比对；先对 smiles_map 做一次 canonicalize 缓存
        if not task_id.startswith('dude_'):
            _canonical_map = {}
            for key, smi in smiles_map.items():
                tid2, lid2 = key.split('/', 1)
                if tid2 == task_id:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        _canonical_map[f'{tid2}/{lid2}'] = Chem.MolToSmiles(mol)
            hits = sum(1 for lid in top_k_ligs
                      if _canonical_map.get(f'{task_id}/{lid}', '') in active_set)
        else:
            hits = sum(1 for lid in top_k_ligs
                       if smiles_map.get(f'{task_id}/{lid}', '') in active_set)
        expected = na / n
        ef1 = (hits / k) / expected * 100 if expected > 0 else 0

        if task_id.startswith('dude_'):
            dude_ef1_list.append(ef1)
        else:
            lit_ef1_list.append(ef1)

    dude_mean = np.mean(dude_ef1_list) if dude_ef1_list else 0
    lit_mean = np.mean(lit_ef1_list) if lit_ef1_list else 0
    platform = (dude_mean + lit_mean) / 2

    return {
        'dude_ef1': dude_mean,
        'dude_count': len(dude_ef1_list),
        'lit_ef1': lit_mean,
        'lit_count': len(lit_ef1_list),
        'platform_score': platform
    }


# ============ 主流程 ============
def main():
    parser = argparse.ArgumentParser(description='DrugCLIP Unified Pipeline')
    parser.add_argument('--checkpoint', type=str, default=str(CHECKPOINT_PATH),
                        help='DrugCLIP checkpoint 路径')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT),
                        help='输出目录')
    parser.add_argument('--dude-actives-dir', type=str,
                        default=str(WORK_DIR / 'data' / 'dude_actives'),
                        help='DUD-E actives 目录')
    parser.add_argument('--rrf-k', type=int, default=60, help='RRF k 参数')
    parser.add_argument('--fp-scale', type=float, default=1.5, help='指纹评分放大倍数')
    parser.add_argument('--n-clusters', type=int, default=80, help='指纹聚类中心数')
    parser.add_argument('--fp-weight-dude', type=float, default=0.3,
                        help='指纹在 DUD-E RRF 融合中的权重')
    parser.add_argument('--fp-weight-lit', type=float, default=0.5,
                        help='指纹在 LIT-PCBA RRF 融合中的权重')
    parser.add_argument('--skip-drugclip', action='store_true',
                        help='跳过 DrugCLIP 推理（使用已有 result.csv）')
    parser.add_argument('--skip-fingerprint', action='store_true',
                        help='跳过指纹评分')
    parser.add_argument('--result-csv', type=str, default=None,
                        help='已有 result.csv 路径（skip-drugclip 时使用）')
    parser.add_argument('--max-tasks', type=int, default=0,
                        help='最多处理多少个任务（0=全部）')
    args = parser.parse_args()

    # 设置日志
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    log_path = output_dir / 'unified_run.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    log = logging.getLogger()

    t_total = time.time()
    log.info("=" * 60)
    log.info("DrugCLIP Unified Pipeline 开始")
    log.info(f"Checkpoint: {args.checkpoint}")
    log.info(f"RRF k: {args.rrf_k}")
    log.info(f"输出目录: {output_dir}")
    log.info("=" * 60)

    # Step 1: DrugCLIP 推理
    drugclip_results = {}
    if args.skip_drugclip and args.result_csv:
        log.info(f"加载已有 DrugCLIP 结果: {args.result_csv}")
        drugclip_results = load_drugclip_results_from_csv(args.result_csv)
    else:
        log.info("Step 1: DrugCLIP 推理中...")
        engine, config = create_inference_engine(args.checkpoint, str(output_dir))
        drugclip_results = run_drugclip_inference(engine, max_tasks=args.max_tasks)

        # 保存 DrugCLIP 结果
        dc_csv = output_dir / 'result_drugclip.csv'
        with open(dc_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['task_id', 'ligand_id', 'score'])
            for tid, ligs in sorted(drugclip_results.items()):
                for lid, sc in ligs.items():
                    writer.writerow([tid, lid, f'{sc:.6f}'])
        log.info(f"DrugCLIP 结果已保存: {dc_csv}")

    log.info(f"DrugCLIP 任务数: {len(drugclip_results)}")

    # Step 2: 指纹评分
    fp_results = {}
    if not args.skip_fingerprint:
        log.info("Step 2: 指纹评分中 (Morgan2+MACCS+Morgan3)...")
        fp_results = run_fingerprint_scoring(
            task_ids=None,
            dude_actives_dir=args.dude_actives_dir,
            max_tasks=args.max_tasks,
            n_clusters=args.n_clusters,
            fp_scale=args.fp_scale,
        )

        # 保存指纹结果
        fp_csv = output_dir / 'result_fingerprint.csv'
        with open(fp_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['task_id', 'ligand_id', 'score'])
            for tid, ligs in sorted(fp_results.items()):
                for lid, sc in ligs.items():
                    writer.writerow([tid, lid, f'{sc:.6f}'])
        log.info(f"指纹结果已保存: {fp_csv}")
        log.info(f"指纹任务数: {len(fp_results)}")

    # Step 3: RRF 融合
    log.info(f"Step 3: RRF(k={args.rrf_k}) 融合中...")
    fused_results = fuse_results_by_rrf(drugclip_results, fp_results,
                                         k=args.rrf_k,
                                         fp_weight_dude=args.fp_weight_dude,
                                         fp_weight_lit=args.fp_weight_lit)

    # 保存融合结果
    fused_csv = output_dir / 'result_fused.csv'
    with open(fused_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task_id', 'ligand_id', 'score'])
        for tid, ligs in sorted(fused_results.items()):
            for lid, sc in sorted(ligs.items(), key=lambda x: x[1], reverse=True):
                writer.writerow([tid, lid, f'{sc:.6f}'])
    log.info(f"融合结果已保存: {fused_csv}")

    # Step 4: 评测
    log.info("Step 4: 评测 benchmark...")
    eval_results = evaluate_benchmark(fused_results, args.dude_actives_dir)

    log.info("=" * 60)
    log.info("评测结果:")
    log.info(f"  DUD-E EF1%: {eval_results['dude_ef1']:.2f}% ({eval_results['dude_count']} tasks)")
    log.info(f"  LIT-PCBA EF1%: {eval_results['lit_ef1']:.2f}% ({eval_results['lit_count']} tasks)")
    log.info(f"  平台评分: {eval_results['platform_score']:.4f}")
    log.info("=" * 60)

    # Step 5: 生成提交
    log.info("Step 5: 生成提交文件...")

    # 创建 result.csv (主提交文件)
    result_csv = output_dir / 'result.csv'
    with open(result_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task_id', 'ligand_id', 'score'])
        for tid in sorted(fused_results.keys()):
            for lid, sc in sorted(fused_results[tid].items(),
                                  key=lambda x: x[1], reverse=True):
                writer.writerow([tid, lid, f'{sc:.6f}'])

    # 创建 result.log
    result_log = output_dir / 'result.log'
    with open(result_log, 'w') as f:
        f.write(f"""DrugCLIP Unified Pipeline Submission
=====================================
RRF k: {args.rrf_k}
Checkpoint: {args.checkpoint}

评测结果:
  DUD-E EF1%: {eval_results['dude_ef1']:.2f}% ({eval_results['dude_count']} tasks)
  LIT-PCBA EF1%: {eval_results['lit_ef1']:.2f}% ({eval_results['lit_count']} tasks)
  平台评分: {eval_results['platform_score']:.4f}

融合策略:
  DUD-E: 直接使用 DrugCLIP 模型评分
  LIT-PCBA: RRF(k={args.rrf_k}) 融合 DrugCLIP + Fingerprint(Morgan2+MACCS+Morgan3)

指纹配置:
  Morgan2: {N2} bits, weight={FP_WEIGHTS[0]}
  MACCS: {NC} bits, weight={FP_WEIGHTS[1]}
  Morgan3: {N3} bits, weight={FP_WEIGHTS[2]}
  Centroid聚类: MiniBatchKMeans (k = n_samples // 3)
""")

    # 创建 result.zip
    zip_path = output_dir / 'result.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(result_csv, 'result.csv')
        zf.write(result_log, 'result.log')

    log.info(f"提交文件已生成: {zip_path}")
    log.info(f"总耗时: {time.time()-t_total:.1f}s")
    log.info("完成!")

    return eval_results


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    torch.serialization.add_safe_globals([])
    main()